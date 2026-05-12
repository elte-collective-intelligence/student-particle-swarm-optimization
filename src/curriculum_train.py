import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.distributions as d
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from tensordict.nn.distributions import CompositeDistribution
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from models import DimAgnosticNet, DimAgnosticCritic, build_curriculum_policy
from training.curriculum import CurriculumManager, Stage, make_env
from eval.generalization import GeneralizationEvaluator


# ---------------------------------------------------------------------------
# Eval grid per curriculum type
# Dims/functions marked with * are held-out (never seen during training).
# ---------------------------------------------------------------------------

EVAL_GRIDS = {
    # Trained on 2D, 5D, 10D, 30D → held-out: 3D*, 7D*, 15D*
    "dimension": {
        "dims": [2, 3, 5, 7, 10, 15, 30],
        "functions": ["sphere", "rastrigin"],
    },
    # Trained on sphere, rosenbrock, rastrigin, eggholder (all 2D)
    # held-out: dynamic_sphere*
    "function": {
        "dims": [2],
        "functions": ["sphere", "rosenbrock", "rastrigin", "eggholder", "dynamic_sphere"],
    },
    # Trained on static/slow/fast sphere → held-out: default shift_speed (dynamic_sphere*)
    "dynamics": {
        "dims": [2],
        "functions": ["sphere", "dynamic_sphere"],
    },
    # Combined: rastrigin at 2D/5D/10D → held-out: sphere 7D*, rosenbrock 5D*
    "combined": {
        "dims": [2, 5, 10],
        "functions": ["sphere", "rosenbrock", "rastrigin"],
    },
}


# ---------------------------------------------------------------------------
# Policy / critic builders
# ---------------------------------------------------------------------------


def build_policy(net: DimAgnosticNet, env, device):
    return build_curriculum_policy(net, env, device)


def build_critic(critic_net: DimAgnosticCritic) -> TensorDictModule:
    return TensorDictModule(
        critic_net,
        in_keys=["avg_pos", "avg_vel"],
        out_keys=["state_value"],
    )


# ---------------------------------------------------------------------------
# GAE (adapted from main.py)
# ---------------------------------------------------------------------------


def compute_gae(data, critic, gamma: float, lmbda: float):
    with torch.no_grad():
        data = critic(data)
        values = data["state_value"]
        if values.dim() > 2:
            values = values.squeeze(-1)

        next_data = critic(data["next"])
        next_values = next_data["state_value"]
        if next_values.dim() > 2:
            next_values = next_values.squeeze(-1)

        rewards = data["next"][("agents", "reward")]
        dones = data["next"][("agents", "done")].float()
        deltas = rewards + gamma * next_values * (1 - dones) - values

        if len(data.batch_size) == 2:
            T = data.batch_size[1]
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros_like(deltas[..., 0, :])
            for t in reversed(range(T)):
                gae = deltas[..., t, :] + gamma * lmbda * (1 - dones[..., t, :]) * gae
                advantages[..., t, :] = gae
        else:
            advantages = deltas

        data["advantage"] = advantages
        data["value_target"] = advantages + values
    return data


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss(subdata, policy, critic, cfg) -> torch.Tensor:
    old_log_prob = subdata["old_log_prob"]
    advantages = subdata["advantage"]
    value_targets = subdata["value_target"]

    # ProbabilisticActor stores its inner TensorDictModule in module[0]
    subdata = policy.module[0](subdata)

    inertia_dist = d.Normal(subdata[("params", "inertia", "loc")],
                            subdata[("params", "inertia", "scale")])
    cog_dist = d.Normal(subdata[("params", "cognitive", "loc")],
                        subdata[("params", "cognitive", "scale")])
    soc_dist = d.Normal(subdata[("params", "social", "loc")],
                        subdata[("params", "social", "scale")])

    new_log_prob = (
        inertia_dist.log_prob(subdata["inertia"]).sum(-1)
        + cog_dist.log_prob(subdata["cognitive"]).sum(-1)
        + soc_dist.log_prob(subdata["social"]).sum(-1)
    )
    entropy = (
        inertia_dist.entropy().sum(-1)
        + cog_dist.entropy().sum(-1)
        + soc_dist.entropy().sum(-1)
    ).mean()

    subdata = critic(subdata)
    values = subdata["state_value"].squeeze(-1)

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    surr = torch.min(
        ratio * adv,
        torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv,
    )

    return (
        -surr.mean()
        + 0.5 * ((values - value_targets.detach()) ** 2).mean()
        - cfg.entropy_coef * entropy
    )


# ---------------------------------------------------------------------------
# Stage training
# ---------------------------------------------------------------------------


def train_stage(stage: Stage, net, critic_net, optim, cfg, device, early_stop_fn=None):
    """
    Train shared net/critic_net on one curriculum stage.

    Args:
        early_stop_fn: called with (reward) each iter; return True to stop early.

    Returns:
        List of per-iteration episode rewards.
    """
    env = make_env(
        stage.dim, stage.landscape_name,
        cfg.env.num_agents, cfg.env.batch_size, cfg.env.delta,
        device, stage.landscape_kwargs,
    )
    policy = build_policy(net, env, device)
    critic = build_critic(critic_net)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
    )
    max_steps = max(cfg.frames_per_batch // cfg.env.batch_size, 10)
    rewards = []

    for it in range(stage.max_iters):
        policy.eval()
        with torch.no_grad():
            data = env.rollout(max_steps=max_steps, policy=policy)

        data = compute_gae(data, critic, cfg.gamma, cfg.lmbda)
        old_lp = (
            data["inertia_log_prob"].sum(-1)
            + data["cognitive_log_prob"].sum(-1)
            + data["social_log_prob"].sum(-1)
        )
        data["old_log_prob"] = old_lp
        replay_buffer.extend(data.view(-1))

        policy.train()
        critic_net.train()
        for _ in range(cfg.num_epochs):
            for sub in replay_buffer:
                loss = ppo_loss(sub, policy, critic, cfg)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(critic_net.parameters()),
                    cfg.max_grad_norm,
                )
                optim.step()

        ep_r = data[("agents", "episode_reward")][..., -1].mean().item()
        rewards.append(ep_r)
        replay_buffer.empty()

        print(f"  iter {it+1:3d}/{stage.max_iters}  reward={ep_r:.3f}")

        if early_stop_fn and early_stop_fn(ep_r):
            print("  → threshold reached, advancing")
            break

    return rewards


def eval_on_task(net, stage: Stage, cfg, device, n_episodes: int = 5) -> float:
    """
    Evaluate the current policy on a specific task without any weight updates.
    Returns mean episode reward over n_episodes rollouts.
    Used to measure transfer: how well does the current policy do on the final
    hard task regardless of what stage it is currently training on?
    """
    env = make_env(
        stage.dim, stage.landscape_name,
        cfg.env.num_agents, cfg.env.batch_size, cfg.env.delta,
        device, stage.landscape_kwargs,
    )
    policy = build_policy(net, env, device)
    policy.eval()
    max_steps = max(cfg.frames_per_batch // cfg.env.batch_size, 10)
    rewards = []
    # Use a local generator so we don't corrupt the global training RNG state.
    gen = torch.Generator()
    with torch.no_grad():
        for ep in range(n_episodes):
            gen.manual_seed(42 + ep)
            data = env.rollout(max_steps=max_steps, policy=policy)
            rewards.append(data[("agents", "episode_reward")][..., -1].mean().item())
    return float(np.mean(rewards))


def run_direct_baseline(final_stage: Stage, total_iters: int, cfg, device, seed: int):
    """
    Train a fresh network directly on the hardest task for total_iters iterations.
    Returns the per-iteration reward list on the final task.
    """
    torch.manual_seed(seed + 10000)
    net = DimAgnosticNet(hidden_size=cfg.model.hidden_size).to(device)
    critic_net = DimAgnosticCritic(hidden_size=cfg.model.hidden_size).to(device)
    optim = torch.optim.Adam(
        list(net.parameters()) + list(critic_net.parameters()),
        lr=cfg.model.learning_rate,
    )
    direct_stage = Stage(
        name=f"direct_{final_stage.name}",
        dim=final_stage.dim,
        landscape_name=final_stage.landscape_name,
        max_iters=total_iters,
        threshold=float("-inf"),
        landscape_kwargs=final_stage.landscape_kwargs,
    )
    print(f"\n  [Direct baseline] dim={final_stage.dim}, fn={final_stage.landscape_name}, iters={total_iters}")
    return train_stage(direct_stage, net, critic_net, optim, cfg, device)


def run_domain_randomization_baseline(stages, total_iters: int, cfg, device, seed: int):
    """
    Train a fresh network with domain randomisation: at each iteration one
    stage is selected uniformly at random.  Performance is measured on the
    final hard task after every iteration so it can be plotted on the same
    x-axis as the curriculum and direct baselines.

    Returns: list of length total_iters — reward on the final hard task.
    """
    rng = np.random.default_rng(seed + 20000)
    torch.manual_seed(seed + 20000)

    net = DimAgnosticNet(hidden_size=cfg.model.hidden_size).to(device)
    critic_net = DimAgnosticCritic(hidden_size=cfg.model.hidden_size).to(device)
    optim = torch.optim.Adam(
        list(net.parameters()) + list(critic_net.parameters()),
        lr=cfg.model.learning_rate,
    )

    final_stage = stages[-1]
    max_steps = max(cfg.frames_per_batch // cfg.env.batch_size, 10)
    rewards_on_final = []

    print(f"\n  [Domain-rand baseline] {total_iters} iters over {len(stages)} stages")
    for it in range(total_iters):
        stage = stages[int(rng.integers(len(stages)))]

        env = make_env(
            stage.dim, stage.landscape_name,
            cfg.env.num_agents, cfg.env.batch_size, cfg.env.delta,
            device, stage.landscape_kwargs,
        )
        policy = build_policy(net, env, device)
        critic = build_critic(critic_net)

        policy.eval()
        with torch.no_grad():
            data = env.rollout(max_steps=max_steps, policy=policy)

        data = compute_gae(data, critic, cfg.gamma, cfg.lmbda)
        old_lp = (
            data["inertia_log_prob"].sum(-1)
            + data["cognitive_log_prob"].sum(-1)
            + data["social_log_prob"].sum(-1)
        )
        data["old_log_prob"] = old_lp
        # Fresh buffer each iter: stages have different dims so tensors are
        # incompatible across iterations — LazyTensorStorage can't be reused.
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(cfg.frames_per_batch, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.minibatch_size,
        )
        replay_buffer.extend(data.view(-1))

        policy.train()
        critic_net.train()
        for _ in range(cfg.num_epochs):
            for sub in replay_buffer:
                loss = ppo_loss(sub, policy, critic, cfg)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(critic_net.parameters()),
                    cfg.max_grad_norm,
                )
                optim.step()

        # Eval on the final hard task (no weight update)
        r = eval_on_task(net, final_stage, cfg, device, n_episodes=3)
        rewards_on_final.append(r)
        print(f"  [domain rand] iter {it+1:3d}/{total_iters}  stage={stage.name}  final_r={r:.3f}")

    return rewards_on_final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(transfer_evals_by_seed, direct_rewards_by_seed, domain_rand_by_seed,
                    stage_logs, output_dir):
    """
    Three-way comparison on the SAME final hard task.

    X-axis: total training iterations (cumulative across all curriculum stages).
    Curriculum:    performance on the final hard task after each stage completes.
    Direct:        fresh network trained only on the final hard task.
    Domain rand:   fresh network trained on randomly sampled stages each iter.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    n_seeds = len(transfer_evals_by_seed)

    # --- Curriculum transfer curve ---
    all_xs = sorted({x for evals in transfer_evals_by_seed.values() for x, _ in evals})
    if len(all_xs) >= 2:
        interp_mat = []
        for evals in transfer_evals_by_seed.values():
            xs_s = [x for x, _ in evals]
            ys_s = [y for _, y in evals]
            interp_mat.append(np.interp(all_xs, xs_s, ys_s))
        interp_mat = np.array(interp_mat)
        cmean, cstd = interp_mat.mean(0), interp_mat.std(0)
        ax.plot(all_xs, cmean, color="steelblue", linewidth=2,
                label="Curriculum (eval on final task)", marker="o", markersize=4)
        ax.fill_between(all_xs, cmean - cstd, cmean + cstd, alpha=0.2, color="steelblue")

    # --- Direct baseline curve ---
    if direct_rewards_by_seed:
        dir_arrs = list(direct_rewards_by_seed.values())
        min_d = min(len(a) for a in dir_arrs)
        dir_mat = np.array([a[:min_d] for a in dir_arrs])
        dm, ds = dir_mat.mean(0), dir_mat.std(0)
        xd = np.arange(min_d)
        ax.plot(xd, dm, color="tomato", linewidth=2, label="Direct (final task only)")
        ax.fill_between(xd, dm - ds, dm + ds, alpha=0.2, color="tomato")

    # --- Domain randomization curve ---
    if domain_rand_by_seed:
        dr_arrs = list(domain_rand_by_seed.values())
        min_dr = min(len(a) for a in dr_arrs)
        dr_mat = np.array([a[:min_dr] for a in dr_arrs])
        drm, drs = dr_mat.mean(0), dr_mat.std(0)
        xdr = np.arange(min_dr)
        ax.plot(xdr, drm, color="seagreen", linewidth=2, label="Domain rand (random stage each iter)")
        ax.fill_between(xdr, drm - drs, drm + drs, alpha=0.2, color="seagreen")

    # --- Stage boundaries (mean iteration count across seeds) ---
    if stage_logs and len(stage_logs[0]) > 0:
        n_stages = len(stage_logs[0])
        boundary = 0
        for s_idx in range(n_stages):
            mean_iters = np.mean([log[s_idx]["iters_used"] for log in stage_logs
                                  if s_idx < len(log)])
            boundary += mean_iters
            ax.axvline(boundary, color="gray", linestyle="--", alpha=0.45)
            # Use axes-fraction y so the label is never clipped by data limits.
            ax.text(boundary + 0.3, 0.02, stage_logs[0][s_idx]["stage"],
                    fontsize=7, rotation=90, va="bottom", color="gray",
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel(f"Total training iterations  (n={n_seeds} seeds, mean ± std)")
    ax.set_ylabel("Reward on final hard task")
    ax.set_title("Sample efficiency: curriculum vs. direct vs. domain randomisation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close()


def plot_curriculum_progression(all_records_by_seed, stage_logs, output_dir):
    """
    Separate plot showing the full reward trajectory across all curriculum stages.
    This is NOT the comparison plot — it shows how the training evolved stage by stage.
    """
    arrs = list(all_records_by_seed.values())
    min_len = min(len(a) for a in arrs)
    mat = np.array([a[:min_len] for a in arrs])
    mean, std = mat.mean(0), mat.std(0)
    xs = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, mean, color="steelblue", linewidth=1.5)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color="steelblue")

    if stage_logs and len(stage_logs[0]) > 0:
        n_stages = len(stage_logs[0])
        boundary = 0
        for s_idx in range(n_stages):
            mean_iters = np.mean([log[s_idx]["iters_used"] for log in stage_logs
                                  if s_idx < len(log)])
            boundary += mean_iters
            ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
            ax.text(boundary + 0.3, 0.02, stage_logs[0][s_idx]["stage"],
                    fontsize=7, rotation=90, va="bottom", color="gray",
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel(f"Training iteration  (n={len(arrs)} seeds, mean ± std)")
    ax.set_ylabel("Episode reward (on current stage's task)")
    ax.set_title("Curriculum training progression")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_progression.png"), dpi=150)
    plt.close()


def plot_generalization_matrix(result, output_dir):
    """Heatmap of zero-shot transfer mean rewards."""
    matrix = result["matrix"]
    dims = result["dims"]
    functions = result["functions"]

    fig, ax = plt.subplots(figsize=(max(len(functions) * 1.5, 4), max(len(dims) * 1.2, 3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(functions)))
    ax.set_xticklabels(functions, rotation=30, ha="right")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"{d}D" for d in dims])
    ax.set_title("Generalization matrix — mean reward across seeds")
    plt.colorbar(im, ax=ax)
    for i in range(len(dims)):
        for j in range(len(functions)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generalization.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs/curriculum", config_name="dimension")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(get_original_cwd(), cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seeds = list(cfg.seeds)
    ctype = cfg.curriculum.type
    print(f"Device: {device}  |  Curriculum: {ctype}  |  Seeds: {seeds}")
    print(f"Output: {output_dir}")

    curriculum_rewards_by_seed = {}
    all_stage_rewards_by_seed = {}
    direct_rewards_by_seed = {}
    domain_rand_by_seed = {}
    gen_matrices = []
    all_stage_logs = []

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'#'*60}")

        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Fresh networks per seed (shared across stages within this seed)
        # ------------------------------------------------------------------
        torch.manual_seed(seed)
        net = DimAgnosticNet(hidden_size=cfg.model.hidden_size).to(device)
        critic_net = DimAgnosticCritic(hidden_size=cfg.model.hidden_size).to(device)
        optim = torch.optim.Adam(
            list(net.parameters()) + list(critic_net.parameters()),
            lr=cfg.model.learning_rate,
        )

        curriculum = CurriculumManager.from_cfg(cfg)
        final_stage = curriculum.stages[-1]
        all_records = []
        # transfer_evals: (cumulative_iters, reward_on_final_hard_task)
        # Measured after each stage — this is what we compare to the direct baseline.
        transfer_evals = []
        cumulative_iters = 0

        # Measure random-init performance on the final task (iteration 0)
        init_r = eval_on_task(net, final_stage, cfg, device)
        transfer_evals.append((0, init_r))

        # ------------------------------------------------------------------
        # Curriculum training — eval on final task after every stage
        # ------------------------------------------------------------------
        while not curriculum.done:
            stage = curriculum.current
            print(f"\n{'='*55}")
            print(f"Stage: {stage.name}  dim={stage.dim}  fn={stage.landscape_name}")
            print(f"{'='*55}")

            rewards = train_stage(
                stage, net, critic_net, optim, cfg, device,
                early_stop_fn=curriculum.update,
            )
            cumulative_iters += len(rewards)
            for r in rewards:
                all_records.append({"stage": stage.name, "reward": r})

            # How well does the current policy do on the final hard task?
            final_task_r = eval_on_task(net, final_stage, cfg, device)
            transfer_evals.append((cumulative_iters, final_task_r))
            print(f"  [transfer eval on {final_stage.name}] reward={final_task_r:.3f}")

        print(f"\n{curriculum.summary()}")
        all_stage_logs.append(curriculum.stage_log)
        curriculum_rewards_by_seed[seed] = transfer_evals          # (iter, reward) pairs
        all_stage_rewards_by_seed[seed] = [r["reward"] for r in all_records]

        # ------------------------------------------------------------------
        # Direct baseline comparison (same total iters as curriculum used)
        # ------------------------------------------------------------------
        total_iters = sum(e["iters_used"] for e in curriculum.stage_log)
        if cfg.get("run_baseline_comparison", True):
            direct_rewards = run_direct_baseline(final_stage, total_iters, cfg, device, seed)
            direct_rewards_by_seed[seed] = direct_rewards

        # ------------------------------------------------------------------
        # Domain randomisation baseline (same total iters, random stage each)
        # ------------------------------------------------------------------
        if cfg.get("run_domain_rand", True):
            dr_rewards = run_domain_randomization_baseline(
                curriculum.stages, total_iters, cfg, device, seed
            )
            domain_rand_by_seed[seed] = dr_rewards

        # ------------------------------------------------------------------
        # Save per-seed checkpoint (after baselines so all data is available)
        # ------------------------------------------------------------------
        torch.save(
            {
                "net_state": net.state_dict(),
                "critic_state": critic_net.state_dict(),
                "curriculum_log": curriculum.stage_log,
                "transfer_evals": transfer_evals,
                "stage_rewards": all_stage_rewards_by_seed[seed],
                "direct_rewards": direct_rewards_by_seed.get(seed, []),
                "domain_rand_rewards": domain_rand_by_seed.get(seed, []),
                "seed": seed,
                "model_cfg": {
                    "hidden_size": cfg.model.hidden_size,
                    "num_agents": cfg.env.num_agents,
                    "delta": cfg.env.delta,
                },
            },
            os.path.join(seed_dir, "model.pt"),
        )

        # ------------------------------------------------------------------
        # Zero-shot generalization for this seed
        # ------------------------------------------------------------------
        if cfg.get("run_generalization", True):
            grid = EVAL_GRIDS.get(ctype, EVAL_GRIDS["dimension"])
            eval_max_steps = max(cfg.frames_per_batch // cfg.env.batch_size, 100)
            evaluator = GeneralizationEvaluator(
                net, device,
                num_agents=cfg.env.num_agents,
                batch_size=cfg.env.batch_size,
                delta=cfg.env.delta,
                n_episodes=cfg.get("n_eval_episodes", 5),
                max_steps=eval_max_steps,
            )
            print(f"\n  Zero-shot eval: dims={grid['dims']}, fns={grid['functions']}")
            result = evaluator.generalization_matrix(grid["dims"], grid["functions"])
            gen_matrices.append(result["matrix"])
            np.save(os.path.join(seed_dir, "generalization.npy"), result["matrix"])

    # ------------------------------------------------------------------
    # Aggregate across seeds and save plots
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}\n# AGGREGATING {len(seeds)} SEEDS\n{'#'*60}")

    # Three-way comparison: curriculum / direct / domain randomisation on same task
    plot_comparison(
        curriculum_rewards_by_seed,   # {seed: [(iter, reward), ...]}
        direct_rewards_by_seed,       # {seed: [reward, ...]}
        domain_rand_by_seed,          # {seed: [reward, ...]}
        all_stage_logs,
        output_dir,
    )
    print(f"Comparison plot saved → {output_dir}/comparison.png")

    # Full training progression across all curriculum stages
    if all_stage_rewards_by_seed:
        plot_curriculum_progression(all_stage_rewards_by_seed, all_stage_logs, output_dir)
        print(f"Progression plot saved → {output_dir}/curriculum_progression.png")

    grid = EVAL_GRIDS.get(ctype, EVAL_GRIDS["dimension"])
    if gen_matrices:
        mean_matrix = np.nanmean(gen_matrices, axis=0)
        np.save(os.path.join(output_dir, "generalization.npy"), mean_matrix)
        plot_generalization_matrix(
            {"dims": grid["dims"], "functions": grid["functions"], "matrix": mean_matrix},
            output_dir,
        )
        print(f"Generalization heatmap saved → {output_dir}/generalization.png")

    # ------------------------------------------------------------------
    # Persist all plot data to results.json for offline re-plotting
    # ------------------------------------------------------------------
    results = {
        "curriculum_type": ctype,
        "seeds": seeds,
        "stage_logs": all_stage_logs,
        # {str(seed): [[iter, reward], ...]}
        "curriculum_transfer_evals": {
            str(s): [[int(it), float(r)] for it, r in evals]
            for s, evals in curriculum_rewards_by_seed.items()
        },
        # {str(seed): [reward, ...]} — per-iteration rewards across all stages
        "stage_rewards": {
            str(s): [float(r) for r in rs]
            for s, rs in all_stage_rewards_by_seed.items()
        },
        # {str(seed): [reward, ...]}
        "direct_rewards": {
            str(s): [float(r) for r in rs]
            for s, rs in direct_rewards_by_seed.items()
        },
        "domain_rand_rewards": {
            str(s): [float(r) for r in rs]
            for s, rs in domain_rand_by_seed.items()
        },
        "gen_dims": grid["dims"],
        "gen_functions": grid["functions"],
        "gen_matrices": [m.tolist() for m in gen_matrices] if gen_matrices else [],
        "gen_matrix_mean": np.nanmean(gen_matrices, axis=0).tolist() if gen_matrices else [],
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
