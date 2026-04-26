"""
Diversity metrics for PSO swarm analysis.

Provides three families of metrics computed over batched particle populations:

- **Position diversity**: mean pairwise Euclidean distance between particles.
- **Position spread proxy**: convex-hull volume for 2-D / 3-D swarms; axis-aligned
  bounding-box (AABB) volume used as a fallback for higher-dimensional spaces.
- **Velocity diversity / alignment**: magnitude spread (std of speeds) and mean
  pairwise cosine alignment, which ranges from -1 (perfectly anti-aligned) to
  +1 (perfectly aligned).

All functions accept tensors with a leading *batch* dimension so they integrate
naturally with the batched ``PSOEnv`` environment.

Typical shapes
--------------
    positions  : (batch, n_agents, dim)
    velocities : (batch, n_agents, dim)

Return values always carry the batch dimension, e.g. shape ``(batch,)``.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional scipy import (only needed for convex-hull volume)
# ---------------------------------------------------------------------------
try:
    from scipy.spatial import ConvexHull  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


# ===========================================================================
# 1. Position diversity — mean pairwise distance
# ===========================================================================


def mean_pairwise_distance(positions: torch.Tensor) -> torch.Tensor:
    """Compute the mean pairwise Euclidean distance across all particle pairs.

    Parameters
    ----------
    positions:
        Tensor of shape ``(batch, n_agents, dim)``.

    Returns
    -------
    torch.Tensor
        Scalar per batch element, shape ``(batch,)``.
        Returns ``0`` when ``n_agents < 2``.
    """
    batch, n, dim = positions.shape

    if n < 2:
        return torch.zeros(batch, device=positions.device, dtype=positions.dtype)

    # Pairwise squared distances via broadcasting: (batch, n, 1, dim) - (batch, 1, n, dim)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (batch, n, n, dim)
    dist = diff.norm(dim=-1)  # (batch, n, n)

    # Upper-triangle indices (i < j) — same for every batch element
    triu_idx = torch.triu_indices(n, n, offset=1, device=positions.device)
    pairwise = dist[:, triu_idx[0], triu_idx[1]]  # (batch, n_pairs)

    return pairwise.mean(dim=-1)  # (batch,)


# ===========================================================================
# 2. Position spread proxy — convex-hull volume (2-D/3-D) or AABB fallback
# ===========================================================================


def _convex_hull_volume_single(pts: np.ndarray) -> float:
    """Compute the convex-hull volume for a single set of 2-D or 3-D points.

    Returns ``0.0`` if the hull is degenerate (collinear/coplanar points).
    """
    if pts.shape[0] < pts.shape[1] + 1:
        # Need at least dim+1 points to form a non-degenerate hull
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:  # QhullError or similar
        return 0.0


def _aabb_volume(positions: torch.Tensor) -> torch.Tensor:
    """Axis-aligned bounding-box volume as a spread proxy.

    Parameters
    ----------
    positions:
        Tensor ``(batch, n_agents, dim)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``.
    """
    lo = positions.min(dim=1).values  # (batch, dim)
    hi = positions.max(dim=1).values  # (batch, dim)
    extents = (hi - lo).clamp_min(0.0)  # (batch, dim)
    # Product of extents across dimensions
    return extents.prod(dim=-1)  # (batch,)


def position_spread(
    positions: torch.Tensor,
    use_hull: bool = True,
) -> torch.Tensor:
    """Position spread proxy: convex-hull volume (2-D/3-D) or AABB (≥4-D).

    For 2-D swarms the convex-hull "volume" is actually its *area*; for 3-D it
    is the true volume.  For higher dimensions the axis-aligned bounding-box
    volume is used instead because convex-hull computation becomes prohibitively
    expensive.

    When ``scipy`` is not installed or ``use_hull=False``, the AABB fallback is
    always used regardless of dimensionality.

    Parameters
    ----------
    positions:
        Tensor ``(batch, n_agents, dim)``.
    use_hull:
        Whether to attempt convex-hull computation for dim ∈ {2, 3}.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``, same dtype as *positions* (float32 / float64).
    """
    batch, n, dim = positions.shape

    use_scipy_hull = use_hull and _SCIPY_AVAILABLE and dim in (2, 3)

    if not use_scipy_hull:
        if use_hull and dim in (2, 3) and not _SCIPY_AVAILABLE:
            warnings.warn(
                "scipy is not available; falling back to AABB volume for "
                "position_spread().",
                RuntimeWarning,
                stacklevel=2,
            )
        return _aabb_volume(positions)

    # --- scipy path: iterate over batch dimension ---
    pts_cpu = positions.detach().cpu().numpy()  # (batch, n, dim)
    volumes = np.empty(batch, dtype=np.float64)
    for b in range(batch):
        volumes[b] = _convex_hull_volume_single(pts_cpu[b])

    return torch.tensor(volumes, device=positions.device, dtype=positions.dtype)


# ===========================================================================
# 3. Velocity diversity / alignment
# ===========================================================================


def velocity_speed_std(velocities: torch.Tensor) -> torch.Tensor:
    """Standard deviation of particle speeds within each batch element.

    A high value indicates a heterogeneous swarm where some particles move fast
    and others are nearly stationary; a low value indicates uniform speeds.

    Parameters
    ----------
    velocities:
        Tensor ``(batch, n_agents, dim)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``.
    """
    speeds = velocities.norm(dim=-1)  # (batch, n_agents)
    if speeds.shape[1] < 2:
        return torch.zeros(
            speeds.shape[0], device=velocities.device, dtype=velocities.dtype
        )
    return speeds.std(dim=-1)  # (batch,)


def velocity_alignment(velocities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean pairwise cosine similarity of velocity vectors.

    Ranges from **-1** (perfectly anti-aligned) through **0** (orthogonal /
    random directions) to **+1** (all particles moving in the same direction).

    Parameters
    ----------
    velocities:
        Tensor ``(batch, n_agents, dim)``.
    eps:
        Small constant added to norms to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``.
    """
    batch, n, dim = velocities.shape

    if n < 2:
        return torch.zeros(batch, device=velocities.device, dtype=velocities.dtype)

    norms = velocities.norm(dim=-1, keepdim=True).clamp_min(eps)  # (batch, n, 1)
    unit_vels = velocities / norms  # (batch, n, dim)

    # Gram matrix of cosine similarities
    cosine_mat = torch.bmm(unit_vels, unit_vels.transpose(1, 2))  # (batch, n, n)

    # Mean over upper-triangle pairs only
    triu_idx = torch.triu_indices(n, n, offset=1, device=velocities.device)
    pairwise_cos = cosine_mat[:, triu_idx[0], triu_idx[1]]  # (batch, n_pairs)

    return pairwise_cos.mean(dim=-1)  # (batch,)


# ===========================================================================
# 4. Convenience wrapper — compute all metrics at once
# ===========================================================================


def compute_diversity_metrics(
    positions: torch.Tensor,
    velocities: Optional[torch.Tensor] = None,
    use_hull: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute the full suite of diversity metrics in a single call.

    Parameters
    ----------
    positions:
        Tensor ``(batch, n_agents, dim)``.
    velocities:
        Optional tensor ``(batch, n_agents, dim)``.  When *None*, velocity
        metrics are omitted from the output dictionary.
    use_hull:
        Passed through to :func:`position_spread`.

    Returns
    -------
    dict
        Keys:

        - ``"mean_pairwise_dist"``  – mean pairwise distance (position diversity)
        - ``"position_spread"``     – convex-hull or AABB volume
        - ``"velocity_speed_std"``  – std of particle speeds  (if *velocities* given)
        - ``"velocity_alignment"``  – mean pairwise cosine similarity (if *velocities* given)
    """
    metrics: Dict[str, torch.Tensor] = {}

    metrics["mean_pairwise_dist"] = mean_pairwise_distance(positions)
    metrics["position_spread"] = position_spread(positions, use_hull=use_hull)

    if velocities is not None:
        metrics["velocity_speed_std"] = velocity_speed_std(velocities)
        metrics["velocity_alignment"] = velocity_alignment(velocities)

    return metrics


# ===========================================================================
# 5. Streaming tracker — accumulate metrics over an episode
# ===========================================================================


class DiversityTracker:
    """Accumulates per-step diversity metrics and summarises an episode.

    Usage::

        tracker = DiversityTracker()
        for step in range(max_steps):
            ...
            tracker.update(positions, velocities)

        summary = tracker.summarise()  # dict of lists (one value per step)

    Parameters
    ----------
    use_hull:
        Passed through to :func:`position_spread`.
    batch_index:
        Which element of the batch dimension to record.  Defaults to ``0``.
    """

    def __init__(self, use_hull: bool = True, batch_index: int = 0) -> None:
        self.use_hull = use_hull
        self.batch_index = batch_index
        self._history: Dict[str, list] = {
            "mean_pairwise_dist": [],
            "position_spread": [],
            "velocity_speed_std": [],
            "velocity_alignment": [],
        }

    def reset(self) -> None:
        """Clear accumulated history."""
        for key in self._history:
            self._history[key].clear()

    def update(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Record diversity metrics for the current step.

        Parameters
        ----------
        positions:
            Tensor ``(batch, n_agents, dim)``.
        velocities:
            Optional tensor ``(batch, n_agents, dim)``.

        Returns
        -------
        dict
            Scalar float values for the tracked batch element at this step.
        """
        metrics = compute_diversity_metrics(
            positions, velocities, use_hull=self.use_hull
        )
        step_vals: Dict[str, float] = {}

        for key, tensor in metrics.items():
            val = tensor[self.batch_index].item()
            self._history[key].append(val)
            step_vals[key] = val

        return step_vals

    def summarise(self) -> Dict[str, list]:
        """Return the full history as a dictionary of Python lists.

        Keys without any recorded values are omitted.
        """
        return {k: list(v) for k, v in self._history.items() if v}

    def mean_over_episode(self) -> Dict[str, float]:
        """Return the time-averaged value of each tracked metric."""
        return {k: float(np.mean(v)) for k, v in self._history.items() if v}
