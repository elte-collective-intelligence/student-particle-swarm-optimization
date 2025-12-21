# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tqdm import trange

from src.envs.pso_env import PSOEnv
from src.utils import LandscapeWrapper


# ----------------- ACTOR -----------------
class AgentActor(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      # 9 -> 64
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2), # 64 -> 32
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim) # 32 -> 3 (inertia, cognitive, social)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- CRITIC -----------------
class CentralCritic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),        # 7 -> 128
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),   # 128 -> 64
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)         # 64 -> 1
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------- TRAINING LOOP -----------------
def train_small():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Eggholder function
    def eggholder(x):
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x_i = x_pairs[..., 0]
        x_j = x_pairs[..., 1]
        t1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
        t2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
        return (t1 + t2).sum(dim=-1)

    # Environment
    landscape = LandscapeWrapper(eggholder, dim=2)
    env = PSOEnv(
        landscape_fn=landscape,
        num_agents=6,
        dim=2,
        batch_size=[4],
        device=device,
        max_steps=50
    )

    # Dimensions
    # positions(2) + velocities(2) + pbest_pos(2) + pbest_val(1) + neighbor(2) = 9
    obs_dim = 2 + 2 + 2 + 1 + 2
    # mean pos(2) + std pos(2) + best_fit(1) + fit_std(1) + diversity(1) = 7
    critic_in = 2 + 2 + 1 + 1 + 1

    actor = AgentActor(obs_dim).to(device)
    critic = CentralCritic(critic_in).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    # ----------------- EPISODES -----------------
    for ep in trange(50, desc="Episodes"):

        td = env.reset()

        for step in range(20):

            # --------- BUILD OBSERVATION ---------
            obs = torch.cat([
                td["positions"],                      # (B,N,2)
                td["velocities"],                     # (B,N,2)
                td["personal_best_positions"],        # (B,N,2)
                td["personal_best_values"].unsqueeze(-1),  # (B,N,1)
                td["neighbor_summary"]               # (B,N,2)
            ], dim=-1)                                # (B,N,9)

            B, N, D = obs.shape
            obs_flat = obs.reshape(B * N, D)          # (B*N, 9)

            # --------- ACTOR FOR ENV STEP (NO GRAD) ---------
            with torch.no_grad():
                actions_flat_env = actor(obs_flat)    # (B*N, 3)

            actions = actions_flat_env.reshape(B, N, 3)  # (B, N, 3)

            # --------- STEP ENV ---------
            td = TensorDict({"action": actions}, batch_size=env.batch_size)
            td = env.step(td)
            td = td["next"]   # move to next state

            # --------- CRITIC UPDATE (DETACHED FROM ENV GRAPH) ---------
            # global observation for centralized critic
            pos_mean = env.positions.detach().mean(dim=1)   # (B,2)
            pos_std = env.positions.detach().std(dim=1)     # (B,2)
            best_fit = env.personal_best_values.detach().min(dim=1).values.unsqueeze(-1)  # (B,1)
            fit_std = env.personal_best_values.detach().std(dim=1).unsqueeze(-1)          # (B,1)
            diversity = torch.tensor(
                [[torch.pdist(env.positions[b]).mean().item()]
                 for b in range(env.batch_size[0])],
                device=device
            )  # (B,1)

            global_obs = torch.cat(
                [pos_mean, pos_std, best_fit, fit_std, diversity], dim=-1
            )  # (B,7)

            target = td["reward"].detach().sum(dim=1)  # (B,)

            value_pred = critic(global_obs)            # (B,)
            loss_critic = (value_pred - target).pow(2).mean()

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            # --------- ACTOR UPDATE (NEW GRAPH, ONLY THROUGH ACTOR) ---------
            rewards = td["reward"].detach()           # (B,N)

            # recompute actor output with grad
            actions_actor = actor(obs_flat)           # (B*N,3) with grad

            # simple surrogate: scale action magnitude by reward
            loss_actor = -(rewards.mean() * actions_actor.pow(2).mean())

            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()

            # --------- TERMINATION ---------
            if td["done"].all():
                break

        # --------- METRICS ---------
        #print(ep, env.get_metrics())
        print(f"Episode {ep} finished.")



if __name__ == "__main__":
    train_small()
