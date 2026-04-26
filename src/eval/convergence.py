"""
Convergence metrics for PSO evaluation.

Operates on per-episode convergence curves — lists of best-score values recorded
at each environment step — as returned by ``evaluate_policy()``.

Metrics
-------
- **AUC** (area under the convergence curve):  a higher value means the swarm
  finds good solutions *early*, not only at the end.
- **time_to_Xpct** – first step at which the best score reaches X% of the
  episode maximum (where X ∈ {50, 80, 90, 99}).
- **final_score** – last value in the curve.
- **improvement_rate** – OLS slope of the curve (score / step).
- **plateau_fraction** – proportion of steps where the running maximum did not
  improve (proxy for premature convergence).

All helpers accept a list of curves (one per episode) and return either a list
of per-episode values or aggregate statistics (mean ± std over episodes).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-episode helpers
# ---------------------------------------------------------------------------


def _auc(curve: List[float]) -> float:
    """Normalised area under the convergence curve (trapezoidal rule).

    Normalised by the product (n_steps × max_score) so the result lives in
    [0, 1] when all scores are non-negative.  For negative-score landscapes
    (sphere, rastrigin) the absolute max is used, yielding values still
    interpretable as a relative fraction.

    Returns ``0.0`` for empty or single-point curves.
    """
    if len(curve) < 2:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    area = float(np.trapezoid(arr))
    # Normalise by maximum possible area
    peak = float(np.max(np.abs(arr)))
    if peak == 0.0:
        return 0.0
    return area / ((len(arr) - 1) * peak)


def _time_to_pct(curve: List[float], pct: float) -> Optional[int]:
    """First step where best score reaches *pct* of the improvement in the episode.

    For maximization problems (like our PSOEnv which negates loss), this is the
    first step where:
        score >= initial + pct * (peak - initial)

    Parameters
    ----------
    curve:
        List of best scores (one per step), assumed monotonically non-decreasing.
    pct:
        Fraction of improvement, e.g. ``0.90`` for 90% of the way to the peak.

    Returns
    -------
    int or None
        Step index (0-based) or ``None`` if no improvement was made or
        the threshold was not reached.
    """
    if not curve:
        return None
    arr = np.asarray(curve, dtype=np.float64)
    initial = float(arr[0])
    peak = float(arr.max())

    # If no improvement was made during the episode, we can't define progress
    if peak <= initial:
        return None

    threshold = initial + pct * (peak - initial)
    hits = np.where(arr >= threshold)[0]
    return int(hits[0]) if len(hits) > 0 else None


def _improvement_rate(curve: List[float]) -> float:
    """OLS slope of the best-score curve (score improvement per step)."""
    n = len(curve)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(curve, dtype=np.float64)
    sx, sy = x.sum(), y.sum()
    sxy = (x * y).sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx
    return float((n * sxy - sx * sy) / denom) if denom != 0 else 0.0


def _plateau_fraction(curve: List[float]) -> float:
    """Fraction of steps where the running best did not improve."""
    if len(curve) < 2:
        return 0.0
    arr = np.asarray(curve, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    # A step is a plateau step if running_max[t] == running_max[t-1]
    plateau_steps = np.sum(running_max[1:] == running_max[:-1])
    return float(plateau_steps) / float(len(arr) - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_episode_convergence(curve: List[float]) -> Dict[str, object]:
    """Compute all convergence metrics for a single episode.

    Parameters
    ----------
    curve:
        List of best-score values, one per environment step.

    Returns
    -------
    dict
        Keys: ``auc``, ``time_to_50pct``, ``time_to_80pct``, ``time_to_90pct``,
        ``time_to_99pct``, ``final_score``, ``improvement_rate``,
        ``plateau_fraction``.
    """
    return {
        "auc": _auc(curve),
        "time_to_50pct": _time_to_pct(curve, 0.50),
        "time_to_80pct": _time_to_pct(curve, 0.80),
        "time_to_90pct": _time_to_pct(curve, 0.90),
        "time_to_99pct": _time_to_pct(curve, 0.99),
        "final_score": float(curve[-1]) if curve else float("nan"),
        "improvement_rate": _improvement_rate(curve),
        "plateau_fraction": _plateau_fraction(curve),
    }


def aggregate_convergence_metrics(
    curves: List[List[float]],
) -> Dict[str, object]:
    """Compute aggregate convergence statistics over multiple episode curves.

    Parameters
    ----------
    curves:
        One list of best-score values per episode.

    Returns
    -------
    dict
        For each metric key ``k`` produced by :func:`compute_episode_convergence`
        the output contains ``k_mean`` and ``k_std`` (over episodes).
        Additionally ``mean_curve`` and ``std_curve`` give the per-step mean
        and std of the best-score over episodes (useful for plotting).
    """
    if not curves:
        return {}

    per_episode = [compute_episode_convergence(c) for c in curves]

    result: Dict[str, object] = {}
    scalar_keys = [
        "auc", "time_to_50pct", "time_to_80pct", "time_to_90pct",
        "time_to_99pct", "final_score", "improvement_rate", "plateau_fraction",
    ]
    for key in scalar_keys:
        vals = [ep[key] for ep in per_episode if ep[key] is not None]
        result[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        result[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")

    # Mean convergence curve (pad shorter curves with their last value)
    max_len = max(len(c) for c in curves)
    padded = np.array(
        [c + [c[-1]] * (max_len - len(c)) if c else [0.0] * max_len for c in curves],
        dtype=np.float64,
    )
    result["mean_curve"] = padded.mean(axis=0).tolist()
    result["std_curve"] = padded.std(axis=0).tolist()

    return result
