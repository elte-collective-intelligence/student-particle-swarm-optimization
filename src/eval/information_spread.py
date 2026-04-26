"""
Information spread metrics for PSO swarm analysis.

Tracks how quickly the best-found position (global best) propagates through
the swarm's topology.

Conceptual model
----------------
At each time-step the current **global best score** is ``max`` over all
``personal_best_scores``.  A particle ``i`` is declared *informed* when its
``neighborhood_best_score[i]`` is within tolerance of the global best — i.e.,
the topology has already routed that information into particle ``i``'s
neighbourhood.

When the global best *improves* a **spread event** begins.  The tracker
records the per-step adoption fraction until either every particle is informed
or a new event supersedes the old one.  From each event's adoption curve it
derives:

- ``steps_to_50pct``  – half-adoption time (None if never reached)
- ``steps_to_90pct``  – 90 % adoption time  (None if never reached)
- ``steps_to_full``   – time to 100 % adoption (None if never reached)
- ``adoption_rate``   – mean per-step rise in adoption fraction (OLS slope)

Additionally an *information entropy* curve H(p) = −p log₂p − (1−p) log₂(1−p)
is maintained at every step: H is maximal (=1 bit) when exactly half the swarm
is informed and zero when information is either absent or fully spread.

Topology sensitivity
--------------------
* **Global (gBest)**: every particle's neighbourhood contains all others, so
  adoption jumps to 1.0 in the very step the global best is discovered.
* **Ring / Von-Neumann**: information hops one edge per step, so the adoption
  curve rises gradually — the spread speed directly reflects graph diameter.
* **K-nearest (dynamic)**: topology rewires each step, which can accelerate or
  disrupt spread depending on the recompute interval.

Input tensor shapes (all batched, matching ``PSOEnv``)
------------------------------------------------------
    personal_best_scores     : (batch, n_agents)
    neighborhood_best_scores : (batch, n_agents)

Return shapes from pure functions always include the batch dimension ``(batch,)``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

# ===========================================================================
# 1. Pure adoption-fraction helper
# ===========================================================================


def information_adoption_fraction(
    neighborhood_best_scores: torch.Tensor,
    global_best_score: torch.Tensor,
    rel_tol: float = 1e-6,
    abs_floor: float = 1e-8,
) -> torch.Tensor:
    """Fraction of particles whose neighbourhood best matches the global best.

    A particle is *informed* when::

        neighborhood_best_scores[i] >= global_best_score - threshold

    where ``threshold = rel_tol * |global_best_score|.clamp_min(abs_floor)``.
    This formulation is safe for both positive and negative score values (e.g.,
    the sphere / rastrigin functions defined in ``eval.py`` return negatives).

    Parameters
    ----------
    neighborhood_best_scores:
        Tensor ``(batch, n_agents)``.
    global_best_score:
        Tensor ``(batch,)`` — typically ``personal_best_scores.max(dim=-1).values``.
    rel_tol:
        Relative tolerance for the "equal to global best" check.
    abs_floor:
        Minimum absolute tolerance (avoids division-by-zero near 0).

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``, values in ``[0, 1]``.
    """
    # threshold per batch element, broadcast over agents
    threshold = rel_tol * global_best_score.abs().clamp_min(abs_floor)  # (batch,)
    cutoff = global_best_score - threshold  # (batch,)

    informed = neighborhood_best_scores >= cutoff.unsqueeze(-1)  # (batch, n_agents)
    return informed.float().mean(dim=-1)  # (batch,)


def information_entropy(adoption_fraction: torch.Tensor) -> torch.Tensor:
    """Binary entropy H(p) of the adoption fraction, in bits.

    H = −p log₂p − (1−p) log₂(1−p)

    Returns ``0`` at the boundaries p=0 and p=1 (no uncertainty).

    Parameters
    ----------
    adoption_fraction:
        Tensor of any shape with values in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Same shape as *adoption_fraction*.
    """
    # Clamp only the log *argument* (not p itself) so that the identity
    # 0 * log(anything_finite) = 0 holds even at the p=0 and p=1 boundaries.
    # Clamping p in float32 doesn't help because 1 - 1e-9 rounds to 1.0,
    # making (1-p) = 0.0 and log2(0) = -inf, yielding 0 * -inf = NaN.
    p = adoption_fraction.clamp(0.0, 1.0)
    log_p = p.clamp_min(1e-30).log2()  # -inf-safe
    log_1mp = (1.0 - p).clamp_min(1e-30).log2()
    return -(p * log_p + (1.0 - p) * log_1mp)


# ===========================================================================
# 2. Spread-event post-processing helpers
# ===========================================================================


def _steps_to_threshold(curve: List[float], threshold: float) -> Optional[int]:
    """Return the first index in *curve* where the value reaches *threshold*.

    Returns ``None`` if the threshold is never reached.
    """
    for i, val in enumerate(curve):
        if val >= threshold:
            return i
    return None


def _adoption_rate(curve: List[float]) -> float:
    """OLS slope of the adoption fraction curve (fraction / step).

    Returns ``0.0`` for curves shorter than 2 points.
    """
    n = len(curve)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(curve, dtype=np.float64)
    # slope = (n Σxy − Σx Σy) / (n Σx² − (Σx)²)
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx
    return float((n * sxy - sx * sy) / denom) if denom != 0 else 0.0


def _analyse_spread_event(
    adoption_curve: List[float],
    start_step: int,
) -> Dict:
    """Derive spread-speed statistics from a single event's adoption curve.

    Parameters
    ----------
    adoption_curve:
        Per-step adoption fractions starting from the event discovery step.
    start_step:
        Global step index at which the event began.

    Returns
    -------
    dict with keys: ``start_step``, ``steps_to_50pct``, ``steps_to_90pct``,
    ``steps_to_full``, ``adoption_rate``, ``peak_adoption``.
    """
    return {
        "start_step": start_step,
        "steps_to_50pct": _steps_to_threshold(adoption_curve, 0.50),
        "steps_to_90pct": _steps_to_threshold(adoption_curve, 0.90),
        "steps_to_full": _steps_to_threshold(adoption_curve, 1.0 - 1e-6),
        "adoption_rate": _adoption_rate(adoption_curve),
        "peak_adoption": max(adoption_curve) if adoption_curve else 0.0,
    }


# ===========================================================================
# 3. Stateful tracker
# ===========================================================================


class InformationSpreadTracker:
    """Accumulates per-step information-spread data over a single episode.

    Call :meth:`update` once per environment step with the batch tensors
    available from ``PSOEnv``.  After the episode call :meth:`summarise` to
    retrieve the full analysis.

    Parameters
    ----------
    rel_tol:
        Passed to :func:`information_adoption_fraction`.
    abs_floor:
        Passed to :func:`information_adoption_fraction`.
    batch_index:
        Which batch element to record (default ``0``).

    Example
    -------
    ::

        tracker = InformationSpreadTracker()
        for step in range(max_steps):
            ...
            tracker.update(
                personal_best_scores=base_env.personal_best_scores,
                neighborhood_best_scores=base_env.neighborhood_best_scores,
            )
        summary = tracker.summarise()
    """

    def __init__(
        self,
        rel_tol: float = 1e-6,
        abs_floor: float = 1e-8,
        batch_index: int = 0,
    ) -> None:
        self.rel_tol = rel_tol
        self.abs_floor = abs_floor
        self.batch_index = batch_index
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated history (call before a new episode)."""
        self._step: int = 0
        self._prev_global_best: float = -math.inf

        # Per-step records
        self._steps: List[int] = []
        self._adoption_fractions: List[float] = []
        self._global_bests: List[float] = []
        self._entropies: List[float] = []

        # Event tracking
        self._event_start_step: Optional[int] = None
        self._event_curve: List[float] = []
        self._completed_events: List[Dict] = []

    def update(
        self,
        personal_best_scores: torch.Tensor,
        neighborhood_best_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """Record metrics for the current environment step.

        Parameters
        ----------
        personal_best_scores:
            Tensor ``(batch, n_agents)`` from ``PSOEnv.personal_best_scores``.
        neighborhood_best_scores:
            Tensor ``(batch, n_agents)`` from ``PSOEnv.neighborhood_best_scores``.

        Returns
        -------
        dict
            Scalar float metrics for *this step* and the tracked batch element:

            - ``"adoption_fraction"`` — share of informed particles ∈ [0, 1]
            - ``"information_entropy"`` — binary entropy H(p) ∈ [0, 1] bits
            - ``"global_best"`` — current global best score
            - ``"is_new_best"`` — 1.0 if a new global best was found this step
        """
        b = self.batch_index
        global_best_all = personal_best_scores.max(dim=-1).values  # (batch,)
        frac_all = information_adoption_fraction(
            neighborhood_best_scores,
            global_best_all,
            rel_tol=self.rel_tol,
            abs_floor=self.abs_floor,
        )  # (batch,)
        entropy_all = information_entropy(frac_all)  # (batch,)

        gb_val = global_best_all[b].item()
        frac_val = frac_all[b].item()
        ent_val = entropy_all[b].item()

        is_new_best = gb_val > self._prev_global_best + self.abs_floor

        # ----- Record per-step data -----
        self._steps.append(self._step)
        self._global_bests.append(gb_val)
        self._adoption_fractions.append(frac_val)
        self._entropies.append(ent_val)

        # ----- Event management -----
        if is_new_best:
            # Close out the previous event (if any)
            if self._event_start_step is not None and self._event_curve:
                self._completed_events.append(
                    _analyse_spread_event(self._event_curve, self._event_start_step)
                )
            # Start new event
            self._event_start_step = self._step
            self._event_curve = [frac_val]
            self._prev_global_best = gb_val
        elif self._event_start_step is not None:
            self._event_curve.append(frac_val)

        self._step += 1

        return {
            "adoption_fraction": frac_val,
            "information_entropy": ent_val,
            "global_best": gb_val,
            "is_new_best": float(is_new_best),
        }

    def finalise_episode(self) -> None:
        """Close the last open spread event.

        Call this *after* the final :meth:`update` if you want the last event
        included in :meth:`summarise`.  It is called automatically by
        :meth:`summarise`.
        """
        if self._event_start_step is not None and self._event_curve:
            self._completed_events.append(
                _analyse_spread_event(self._event_curve, self._event_start_step)
            )
            # Prevent double-closing on repeated summarise() calls
            self._event_start_step = None
            self._event_curve = []

    def summarise(self) -> Dict:
        """Return the full episode analysis.

        Returns
        -------
        dict with the following keys:

        ``adoption_fraction_curve`` : List[float]
            Per-step adoption fraction (one value per step).
        ``information_entropy_curve`` : List[float]
            Per-step binary entropy of adoption.
        ``global_best_curve`` : List[float]
            Per-step global best score.
        ``spread_events`` : List[dict]
            One dict per new-global-best event, each containing:
            ``start_step``, ``steps_to_50pct``, ``steps_to_90pct``,
            ``steps_to_full``, ``adoption_rate``, ``peak_adoption``.
        ``num_spread_events`` : int
            Total number of global-best improvement events.
        ``mean_steps_to_50pct`` : Optional[float]
            Average half-adoption time across events (None if no event reached it).
        ``mean_steps_to_90pct`` : Optional[float]
            Average 90 %-adoption time across events (None if no event reached it).
        ``mean_adoption_rate`` : float
            Average OLS adoption-rate slope across events.
        ``mean_peak_adoption`` : float
            Average peak adoption fraction across events.
        ``final_adoption_fraction`` : float
            Adoption fraction at the last recorded step.
        ``total_steps`` : int
            Number of steps recorded.
        """
        self.finalise_episode()

        events = self._completed_events

        def _mean_optional(values: List[Optional[int]]) -> Optional[float]:
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else None

        return {
            "adoption_fraction_curve": list(self._adoption_fractions),
            "information_entropy_curve": list(self._entropies),
            "global_best_curve": list(self._global_bests),
            "spread_events": [dict(e) for e in events],
            "num_spread_events": len(events),
            "mean_steps_to_50pct": _mean_optional(
                [e["steps_to_50pct"] for e in events]
            ),
            "mean_steps_to_90pct": _mean_optional(
                [e["steps_to_90pct"] for e in events]
            ),
            "mean_adoption_rate": (
                float(np.mean([e["adoption_rate"] for e in events])) if events else 0.0
            ),
            "mean_peak_adoption": (
                float(np.mean([e["peak_adoption"] for e in events])) if events else 0.0
            ),
            "final_adoption_fraction": (
                self._adoption_fractions[-1] if self._adoption_fractions else 0.0
            ),
            "total_steps": self._step,
        }

    def mean_over_episode(self) -> Dict[str, float]:
        """Time-averaged scalar metrics (convenient for logging).

        Returns
        -------
        dict with keys ``"mean_adoption_fraction"``, ``"mean_information_entropy"``.
        """
        return {
            "mean_adoption_fraction": (
                float(np.mean(self._adoption_fractions))
                if self._adoption_fractions
                else 0.0
            ),
            "mean_information_entropy": (
                float(np.mean(self._entropies)) if self._entropies else 0.0
            ),
        }
