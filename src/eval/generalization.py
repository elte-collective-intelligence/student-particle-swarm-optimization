import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_curriculum_policy
from training.curriculum import make_env


class GeneralizationEvaluator:
    """
    Evaluates a policy on arbitrary (dim, function) pairs.

    Args:
        net:        DimAgnosticNet instance (weights to evaluate).
        device:     torch device.
        num_agents: number of PSO particles.
        batch_size: environment batch size.
        delta:      neighborhood radius.
        n_episodes: rollouts per condition.
        max_steps:  steps per rollout episode. Should match training rollout
                    length (frames_per_batch // batch_size) so that
                    generalization scores are on the same scale as training
                    rewards. Default 100 is a reasonable minimum; set higher
                    for high-dimensional conditions.
    """

    def __init__(
        self,
        net,
        device: torch.device,
        num_agents: int = 10,
        batch_size: int = 4,
        delta: float = 1.0,
        n_episodes: int = 5,
        max_steps: int = 100,
    ):
        self.net = net
        self.device = device
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.delta = delta
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    def evaluate(
        self,
        dim: int,
        landscape_name: str,
        landscape_kwargs: Optional[dict] = None,
        seed: int = 42,
    ) -> dict:
        """
        Run n_episodes on (dim, landscape_name) and return statistics.

        Returns:
            dict with keys: dim, landscape, mean_reward, std_reward, best_reward
        """
        landscape_kwargs = landscape_kwargs or {}
        env = make_env(dim, landscape_name, self.num_agents, self.batch_size,
                       self.delta, self.device, landscape_kwargs)
        policy = build_curriculum_policy(self.net, env, self.device)

        rewards = []
        policy.eval()
        # Use a local generator so eval does not corrupt the global training RNG.
        gen = torch.Generator()
        with torch.no_grad():
            for ep in range(self.n_episodes):
                gen.manual_seed(seed + ep)
                data = env.rollout(max_steps=self.max_steps, policy=policy)
                ep_r = data[("agents", "episode_reward")][..., -1].mean().item()
                rewards.append(ep_r)

        return {
            "dim": dim,
            "landscape": landscape_name,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_reward": float(np.max(rewards)),
        }

    def generalization_matrix(
        self,
        dims: List[int],
        functions: List[str],
        seed: int = 42,
    ) -> dict:
        """
        Evaluate all (dim, function) combinations.

        Returns:
            dict with keys:
              dims      — list of dimensions tested
              functions — list of function names tested
              matrix    — np.ndarray shape (len(dims), len(functions)) of mean rewards
        """
        matrix = np.full((len(dims), len(functions)), np.nan)
        for i, dim in enumerate(dims):
            for j, fn in enumerate(functions):
                try:
                    result = self.evaluate(dim, fn, seed=seed)
                    matrix[i, j] = result["mean_reward"]
                    print(f"  [{dim}D, {fn}]: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
                except ValueError as e:
                    # Expected for invalid combos (e.g. odd-dim eggholder).
                    print(f"  [{dim}D, {fn}]: skipped — {e}")
                except Exception:
                    # Unexpected error: print full traceback so bugs are visible.
                    print(f"  [{dim}D, {fn}]: FAILED")
                    traceback.print_exc()

        return {"dims": dims, "functions": functions, "matrix": matrix}
