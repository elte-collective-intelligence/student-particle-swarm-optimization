# src/ppo_main.py
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt

from tensordict import TensorDict
from src.envs.pso_env import PSOEnv
from src.utils import LandscapeWrapper


# =========================
#  Actor & Critic Networks
# =========================

class ActorNet(nn.Module):
    """Decentralized actor"""
    def __init__(self, obs_dim, hidden=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim * 2),
        )
        self.action_dim = action_dim

    def forward(self, obs):
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    """Centralized critic"""
    def __init__(self, global_obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
#  Observation Builders
# =========================

def build_local_obs(td):
    obs = torch.cat([
        td["positions"],
        td["velocities"],
        td["personal_best_positions"],
        td["personal_best_values"].unsqueeze(-1),
        td["neighbor_summary"],
    ], dim=-1)
    return obs


def build_global_obs(env):
    B = env.batch_size[0]

    pos_mean = env.positions.detach().mean(dim=1)
    pos_std = env.positions.detach().std(dim=1)
    best_fit = env.personal_best_values.detach().min(dim=1).values.unsqueeze(-1)
    fit_std = env.personal_best_values.detach().std(dim=1).unsqueeze(-1)

    diversity_vals = []
    for b in range(B):
        if env.positions[b].shape[0] > 1:
            d = torch.pdist(env.positions[b]).mean()
        else:
            d = torch.tensor(0.0, device=env.device)
        diversity_vals.append(d)

    diversity = torch.stack(diversity_vals).unsqueeze(-1)
    global_obs = torch.cat([pos_mean, pos_std, best_fit, fit_std, diversity], dim=-1)
    return global_obs


# =========================
#  GAE
# =========================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T, B = rewards.shape
    advantages = torch.zeros(T, B, device=rewards.device)
    gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# =========================
#  PPO-CTDE Training
# =========================

def train_ppo_ctde():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Eggholder Landscape -----
    def eggholder(x):
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x_i = x_pairs[..., 0]
        x_j = x_pairs[..., 1]
        t1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
        t2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
        return (t1 + t2).sum(dim=-1)

    landscape = LandscapeWrapper(eggholder, dim=2)

    env = PSOEnv(
        landscape_fn=landscape,
        num_agents=6,
        dim=2,
        batch_size=[4],
        device=device,
        dynamic=True
    )

    obs_dim = 9
    global_obs_dim = 7
    action_dim = 3

    actor = ActorNet(obs_dim).to(device)
    critic = CriticNet(global_obs_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    clip_eps = 0.2
    ppo_epochs = 4
    minibatch_size = 128
    gamma = 0.99
    lam = 0.95
    max_steps_per_episode = 20
    num_episodes = 50

    all_best = []
    all_div = []

    for ep in trange(num_episodes, desc="PPO Episodes"):
        td = env.reset()

        local_obs_buf = []
        actions_buf = []
        logp_buf = []
        rewards_env_buf = []
        values_env_buf = []
        dones_env_buf = []

        for _ in range(max_steps_per_episode):
            state_td = td["next"] if "next" in td.keys() else td
            local_obs = build_local_obs(state_td)

            B, N, D = local_obs.shape
            obs_flat = local_obs.reshape(B * N, D).to(device)

            with torch.no_grad():
                mean, std = actor(obs_flat)
                dist = torch.distributions.Normal(mean, std)
                actions_flat = dist.sample()
                logp_flat = dist.log_prob(actions_flat).sum(dim=-1)

                global_obs = build_global_obs(env).to(device)
                values_env = critic(global_obs)

            actions = actions_flat.reshape(B, N, action_dim)

            step_td = TensorDict({"action": actions}, batch_size=env.batch_size)
            next_td = env.step(step_td)

            rewards_env = next_td["next", "reward"].mean(dim=1)
            dones_env = next_td["next", "done"].squeeze(-1)

            local_obs_buf.append(obs_flat.cpu())
            actions_buf.append(actions_flat.cpu())
            logp_buf.append(logp_flat.cpu())
            rewards_env_buf.append(rewards_env.cpu())
            values_env_buf.append(values_env.cpu())
            dones_env_buf.append(dones_env.cpu())

            td = next_td
            if dones_env.all():
                break

        with torch.no_grad():
            global_obs_last = build_global_obs(env).to(device)
            last_v_env = critic(global_obs_last)
        values_env_buf.append(last_v_env.cpu())

        rewards_env_tensor = torch.stack(rewards_env_buf)
        values_env_tensor = torch.stack(values_env_buf)
        dones_env_tensor = torch.stack(dones_env_buf)

        advantages_env, returns_env = compute_gae(
            rewards_env_tensor,
            values_env_tensor,
            dones_env_tensor,
            gamma,
            lam,
        )

        advantages_env = (advantages_env - advantages_env.mean()) / (advantages_env.std() + 1e-8)

        B = env.batch_size[0]
        T = len(rewards_env_buf)
        num_agents = env.num_agents

        advantages_agents = advantages_env.unsqueeze(-1).expand(T, B, num_agents).reshape(-1)
        returns_agents = returns_env.unsqueeze(-1).expand(T, B, num_agents).reshape(-1)

        obs_all = torch.stack(local_obs_buf).reshape(-1, obs_dim).to(device)
        act_all = torch.stack(actions_buf).reshape(-1, action_dim).to(device)
        logp_old_all = torch.stack(logp_buf).reshape(-1).to(device)
        adv_all = advantages_agents.to(device)
        ret_all = returns_agents.to(device)

        TBN = obs_all.shape[0]

        for _ in range(ppo_epochs):
            idx = torch.randperm(TBN)
            for start in range(0, TBN, minibatch_size):
                batch_idx = idx[start:start + minibatch_size]

                batch_obs = obs_all[batch_idx]
                batch_act = act_all[batch_idx]
                batch_logp_old = logp_old_all[batch_idx]
                batch_adv = adv_all[batch_idx]
                batch_ret = ret_all[batch_idx]

                mean, std = actor(batch_obs)
                dist = torch.distributions.Normal(mean, std)
                logp_new = dist.log_prob(batch_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (logp_new - batch_logp_old).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy

                value_pred = critic(batch_obs[:, :global_obs_dim])
                critic_loss = (value_pred - batch_ret).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                loss.backward()
                actor_opt.step()
                critic_opt.step()

        metrics = env.get_metrics()
        all_best.append(metrics["best_fitness"])
        all_div.append(metrics["diversity"])
        print(f"Episode {ep}: best_fitness={metrics['best_fitness']:.3f}, diversity={metrics['diversity']:.3f}")

    # ===============================
    # SAVE RESULTS TO UNIQUE FOLDER
    # ===============================

    run_name = datetime.now().strftime("dynamic_%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(all_best)
    plt.title("Best Fitness over PPO Episodes (Dynamic)")
    plt.grid(True)
    plt.savefig(output_dir / "ppo_best_fitness.png", dpi=200)

    plt.figure()
    plt.plot(all_div)
    plt.title("Diversity over PPO Episodes (Dynamic)")
    plt.grid(True)
    plt.savefig(output_dir / "ppo_diversity.png", dpi=200)

    print(f"✅ PPO-CTDE dynamic training finished. Plots saved to: {output_dir}")


if __name__ == "__main__":
    train_ppo_ctde()
