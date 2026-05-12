import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.env import PSOEnv
from torchrl.envs import TransformedEnv, RewardSum


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    dim: int
    landscape_name: str
    max_iters: int
    threshold: float           # rolling-mean reward to advance early; -inf disables
    landscape_kwargs: dict = field(default_factory=dict)
    min_iters: int = 0         # must train at least this many iters before threshold check


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------


class CurriculumManager:
    """
    Manages a sequence of training stages.

    Usage:
        mgr = CurriculumManager.dimension()
        while not mgr.done:
            stage = mgr.current
            # ... train on stage.dim / stage.landscape_name ...
            advanced = mgr.update(episode_reward)
    """

    def __init__(self, stages: List[Stage], window: int = 10):
        self.stages = stages
        self.window = window
        self._idx = 0
        self._iters = 0
        self._history: List[float] = []
        self.stage_log: List[dict] = []

    # ------------------------------------------------------------------
    # Preset curricula
    # ------------------------------------------------------------------
    @classmethod
    def dimension(cls, window: int = 10) -> "CurriculumManager":
        """Grow search-space dimension on Sphere: 2D → 5D → 10D → 30D.
        Sphere is used throughout so dimension is the only changing variable,
        isolating its effect from function complexity.
        """
        return cls(
            [
                Stage("sphere_2d",  2,  "sphere", max_iters=50,  threshold=0.60, min_iters=20),
                Stage("sphere_5d",  5,  "sphere", max_iters=80,  threshold=float("-inf"), min_iters=30),
                Stage("sphere_10d", 10, "sphere", max_iters=100, threshold=float("-inf")),
                Stage("sphere_30d", 30, "sphere", max_iters=120, threshold=float("-inf")),
            ],
            window=window,
        )

    @classmethod
    def function(cls, window: int = 10) -> "CurriculumManager":
        """Increase function complexity in 2D: Sphere → Rosenbrock → Rastrigin → Eggholder."""
        return cls(
            [
                Stage("sphere_2d",     2, "sphere",     max_iters=40,  threshold=1.0,          min_iters=20),
                Stage("rosenbrock_2d", 2, "rosenbrock", max_iters=60,  threshold=float("-inf"), min_iters=25),
                Stage("rastrigin_2d",  2, "rastrigin",  max_iters=80,  threshold=float("-inf"), min_iters=30),
                Stage("eggholder_2d",  2, "eggholder",  max_iters=100, threshold=float("-inf")),
            ],
            window=window,
        )

    @classmethod
    def dynamics(cls, window: int = 10) -> "CurriculumManager":
        """Increase landscape non-stationarity: static → slow → fast dynamics."""
        return cls(
            [
                Stage("static_2d",   2, "sphere",         max_iters=40, threshold=1.0,           min_iters=20),
                Stage("slow_dyn_2d", 2, "dynamic_sphere", max_iters=60, threshold=0.5,           min_iters=25, landscape_kwargs={"shift_speed": 0.05}),
                Stage("fast_dyn_2d", 2, "dynamic_sphere", max_iters=80, threshold=float("-inf"), landscape_kwargs={"shift_speed": 0.3}),
            ],
            window=window,
        )

    @classmethod
    def combined(cls, window: int = 10) -> "CurriculumManager":
        """Function complexity then dimension scaling: sphere 2D → rastrigin 2D → rastrigin 5D → rastrigin 10D."""
        return cls(
            [
                Stage("sphere_2d",     2,  "sphere",    max_iters=40,  threshold=1.0,          min_iters=20),
                Stage("rastrigin_2d",  2,  "rastrigin", max_iters=60,  threshold=float("-inf"), min_iters=25),
                Stage("rastrigin_5d",  5,  "rastrigin", max_iters=80,  threshold=float("-inf"), min_iters=30),
                Stage("rastrigin_10d", 10, "rastrigin", max_iters=100, threshold=float("-inf")),
            ],
            window=window,
        )

    @classmethod
    def from_cfg(cls, cfg) -> "CurriculumManager":
        """Build a curriculum from a Hydra config (reads cfg.curriculum.type and .window)."""
        presets = {
            "dimension": cls.dimension,
            "function": cls.function,
            "dynamics": cls.dynamics,
            "combined": cls.combined,
        }
        ctype = cfg.curriculum.type
        if ctype not in presets:
            raise ValueError(f"Unknown curriculum type: '{ctype}'. Choose from: {list(presets)}")
        return presets[ctype](window=cfg.curriculum.window)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def current(self) -> Stage:
        return self.stages[self._idx]

    @property
    def done(self) -> bool:
        return self._idx >= len(self.stages)

    def update(self, reward: float) -> bool:
        """
        Record one iteration's reward. Returns True if stage just advanced.
        Call once per training iteration.
        """
        self._iters += 1
        self._history.append(reward)

        if self._should_advance():
            self.stage_log.append(
                {
                    "stage": self.current.name,
                    "iters_used": self._iters,
                    "final_reward": float(np.mean(self._history[-5:])),
                }
            )
            self._idx += 1
            self._iters = 0
            self._history = []
            return True
        return False

    def _should_advance(self) -> bool:
        if self._iters >= self.current.max_iters:
            return True
        # threshold == -inf is used as a sentinel to disable reward-based early advance
        past_min = self._iters >= self.current.min_iters
        if past_min and self.current.threshold != float("-inf") and len(self._history) >= self.window:
            if np.mean(self._history[-self.window :]) >= self.current.threshold:
                return True
        return False

    def summary(self) -> str:
        lines = ["Curriculum progression:"]
        for entry in self.stage_log:
            lines.append(
                f"  {entry['stage']:20s} — {entry['iters_used']} iters, "
                f"reward={entry['final_reward']:.3f}"
            )
        if not self.done:
            lines.append(f"  (stopped at stage: {self.current.name})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment factory (shared by curriculum_train and generalization)
# ---------------------------------------------------------------------------


def make_env(
    dim: int,
    landscape_name: str,
    num_agents: int,
    batch_size: int,
    delta: float,
    device: torch.device,
    landscape_kwargs: Optional[dict] = None,
) -> TransformedEnv:
    """Create a wrapped PSOEnv for a given (dim, landscape) combination."""
    from envs.dynamic_functions import DynamicSphere, DynamicRastrigin
    from utils import LandscapeWrapper

    landscape_kwargs = landscape_kwargs or {}

    from envs.functions import STATIC_FUNCTIONS as static

    if landscape_name in static:
        landscape = LandscapeWrapper(static[landscape_name], dim=dim)
    elif landscape_name == "dynamic_sphere":
        landscape = DynamicSphere(dim=dim, **landscape_kwargs)
    elif landscape_name == "dynamic_rastrigin":
        landscape = DynamicRastrigin(dim=dim, **landscape_kwargs)
    else:
        available = list(static.keys()) + ["dynamic_sphere", "dynamic_rastrigin"]
        raise ValueError(f"Unknown landscape '{landscape_name}'. Available: {available}")

    env = PSOEnv(
        landscape=landscape,
        num_agents=num_agents,
        device=device,
        batch_size=(batch_size,),
        delta=delta,
    )
    return TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        device=device,
    )
