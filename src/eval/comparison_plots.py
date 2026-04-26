"""
Comparison plots for multi-topology PSO evaluation.

Produces four publication-quality figure sets from the ``all_results`` list
returned by ``eval_multi_topology.main()``, or from the saved JSON/episode-CSV.

Plot types
----------
1. **Convergence curves** – mean ± std best-score over steps, one panel per
   policy, one line per topology.
2. **Diversity over time** – mean pairwise distance (and optionally velocity
   alignment) averaged across episodes, by topology.
3. **Final fitness distributions** – violin + strip plots of per-episode final
   scores grouped by topology and policy.
4. **Summary heatmap** – topology × metric matrix with colour-coded z-scores
   plus an optional radar / spider chart.

Public API
----------
    from eval.comparison_plots import ComparisonPlotter
    plotter = ComparisonPlotter(results, output_dir)
    plotter.plot_all()

Or use the standalone helpers:
    plot_convergence_curves(results, output_dir)
    plot_diversity_over_time(results, output_dir)
    plot_fitness_distributions(results, output_dir)
    plot_summary_heatmap(results, output_dir)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4E79A7",  # global   – steel blue
    "#F28E2B",  # ring     – amber
    "#E15759",  # von_neumann – tomato
    "#76B7B2",  # knearest – teal
    "#59A14F",  # extra
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
]

_POLICY_LINESTYLE = {
    "Trained": "-",
    "Random": "--",
}

_POLICY_MARKER = {
    "Trained": "o",
    "Random": "s",
}

_FONT_TITLE = 13
_FONT_LABEL = 11
_FONT_TICK = 9
_FONT_LEGEND = 9
_DPI = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _color_map(topologies: List[str]) -> Dict[str, str]:
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(topologies)}


def _topology_order(results: list) -> List[str]:
    return list(dict.fromkeys(r["topology"] for r in results))


def _policy_order(results: list) -> List[str]:
    return list(dict.fromkeys(r["policy"] for r in results))


def _apply_style(ax, xlabel="", ylabel="", title="") -> None:
    ax.set_xlabel(xlabel, fontsize=_FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FONT_LABEL)
    ax.set_title(title, fontsize=_FONT_TITLE, fontweight="bold", pad=8)
    ax.tick_params(labelsize=_FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Saved plot : {path}")


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        warnings.warn("matplotlib not available — skipping plots.", RuntimeWarning)
        return None


# ===========================================================================
# 1. Convergence curves
# ===========================================================================


def plot_convergence_curves(
    results: list,
    output_dir: str,
    filename: str = "convergence_curves.png",
) -> Optional[str]:
    """Mean ± std best-score convergence curves, one panel per policy.

    Parameters
    ----------
    results:
        List of result dicts from ``eval_multi_topology.main()``.  Each dict
        must contain ``convergence_mean_curve`` and ``convergence_std_curve``
        lists.
    output_dir:
        Directory to write the figure into.
    filename:
        Output file name.

    Returns
    -------
    str or None
        Absolute path to the saved figure, or ``None`` if matplotlib is unavailable.
    """
    plt = _get_plt()
    if plt is None:
        return None

    topologies = _topology_order(results)
    policies = _policy_order(results)
    cmap = _color_map(topologies)

    ncols = len(policies)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
    fig.suptitle(
        "Convergence Curves by Topology",
        fontsize=_FONT_TITLE + 1,
        fontweight="bold",
        y=1.02,
    )

    for col, pol in enumerate(policies):
        ax = axes[0][col]
        pol_res = [r for r in results if r["policy"] == pol]

        for res in pol_res:
            topo = res["topology"]
            mean_c = res.get("convergence_mean_curve", [])
            std_c = res.get("convergence_std_curve", [])
            if not mean_c:
                continue
            steps = np.arange(len(mean_c))
            color = cmap[topo]
            ax.plot(
                steps,
                mean_c,
                color=color,
                linewidth=2,
                linestyle=_POLICY_LINESTYLE.get(pol, "-"),
                label=topo,
                zorder=3,
            )
            if std_c:
                lo = np.array(mean_c) - np.array(std_c)
                hi = np.array(mean_c) + np.array(std_c)
                ax.fill_between(steps, lo, hi, color=color, alpha=0.12, zorder=2)

        _apply_style(ax, xlabel="Step", ylabel="Best score", title=f"Policy: {pol}")
        ax.legend(fontsize=_FONT_LEGEND, framealpha=0.85)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save(fig, path)
    return path


# ===========================================================================
# 2. Diversity over time
# ===========================================================================


def _extract_diversity_curves(
    results: list,
    metric: str = "mean_pairwise_dist",
) -> Dict[Tuple[str, str], List[float]]:
    """
    For each (topology, policy) pair return the mean-over-episodes per-step
    diversity curve.

    Curves are extracted from ``episode_histories[ep]["diversity"][metric]``
    (each itself a list of per-step floats).
    """
    curves: Dict[Tuple[str, str], List[float]] = {}
    for res in results:
        key = (res["topology"], res["policy"])
        histories = res.get("episode_histories", [])
        ep_curves = [
            h["diversity"][metric]
            for h in histories
            if "diversity" in h and metric in h["diversity"]
        ]
        if not ep_curves:
            continue
        max_len = max(len(c) for c in ep_curves)
        padded = np.array(
            [c + [c[-1]] * (max_len - len(c)) if c else [] for c in ep_curves if c],
            dtype=np.float64,
        )
        if padded.size == 0:
            continue
        curves[key] = padded.mean(axis=0).tolist()
    return curves


def plot_diversity_over_time(
    results: list,
    output_dir: str,
    filename: str = "diversity_over_time.png",
    metrics: Optional[List[str]] = None,
) -> Optional[str]:
    """Diversity metrics averaged across episodes plotted over time steps.

    Parameters
    ----------
    results:
        List of result dicts.
    output_dir:
        Output directory.
    filename:
        Output file name.
    metrics:
        Diversity metrics to plot.  Defaults to
        ``["mean_pairwise_dist", "velocity_alignment"]``.

    Returns
    -------
    str or None
    """
    plt = _get_plt()
    if plt is None:
        return None

    if metrics is None:
        metrics = ["mean_pairwise_dist", "velocity_alignment"]

    _metric_labels = {
        "mean_pairwise_dist": "Mean pairwise distance",
        "position_spread": "Position spread (AABB vol.)",
        "velocity_alignment": "Velocity alignment (cos sim)",
        "velocity_speed_std": "Velocity speed std",
    }

    topologies = _topology_order(results)
    policies = _policy_order(results)
    cmap = _color_map(topologies)

    nrows = len(metrics)
    ncols = len(policies)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False
    )
    fig.suptitle(
        "Diversity Metrics Over Time",
        fontsize=_FONT_TITLE + 1,
        fontweight="bold",
        y=1.02,
    )

    for row, met in enumerate(metrics):
        label = _metric_labels.get(met, met)
        curves = _extract_diversity_curves(results, metric=met)

        for col, pol in enumerate(policies):
            ax = axes[row][col]
            for topo in topologies:
                key = (topo, pol)
                curve = curves.get(key)
                if curve is None:
                    continue
                steps = np.arange(len(curve))
                ax.plot(
                    steps,
                    curve,
                    color=cmap[topo],
                    linewidth=2,
                    linestyle=_POLICY_LINESTYLE.get(pol, "-"),
                    label=topo,
                    zorder=3,
                )

            title = f"{pol} — {label}" if row == 0 else label
            _apply_style(
                ax, xlabel="Step", ylabel=label, title=title if row == 0 else ""
            )
            if row == 0:
                ax.legend(fontsize=_FONT_LEGEND, framealpha=0.85)
            ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save(fig, path)
    return path


# ===========================================================================
# 3. Final fitness distributions
# ===========================================================================


def plot_fitness_distributions(
    results: list,
    output_dir: str,
    filename: str = "fitness_distributions.png",
) -> Optional[str]:
    """Violin + strip chart of per-episode final scores by topology × policy.

    Parameters
    ----------
    results:
        List of result dicts.  Each must contain ``convergence_per_episode``
        (list of dicts with key ``"final_score"``).
    output_dir:
        Output directory.
    filename:
        Output file name.

    Returns
    -------
    str or None
    """
    plt = _get_plt()
    if plt is None:
        return None

    topologies = _topology_order(results)
    policies = _policy_order(results)
    cmap = _color_map(topologies)

    ncols = len(policies)
    fig, axes = plt.subplots(
        1, ncols, figsize=(max(8, 2.5 * len(topologies)) * ncols, 6), squeeze=False
    )
    fig.suptitle(
        "Final Fitness Score Distributions",
        fontsize=_FONT_TITLE + 1,
        fontweight="bold",
        y=1.02,
    )

    rng = np.random.default_rng(0)  # reproducible jitter

    for col, pol in enumerate(policies):
        ax = axes[0][col]
        pol_res = [r for r in results if r["policy"] == pol]

        all_scores_by_topo: Dict[str, List[float]] = {}
        for res in pol_res:
            topo = res["topology"]
            scores = [
                ep["final_score"]
                for ep in res.get("convergence_per_episode", [])
                if ep.get("final_score") is not None and not np.isnan(ep["final_score"])
            ]
            if scores:
                all_scores_by_topo[topo] = scores

        positions = np.arange(len(topologies))
        violin_data = [all_scores_by_topo.get(t, [0.0]) for t in topologies]
        colors = [cmap[t] for t in topologies]

        # Violin
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.6,
            showmedians=True,
            showextrema=True,
        )
        for i, (pc, col_c) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(col_c)
            pc.set_edgecolor("white")
            pc.set_alpha(0.75)
        for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
            if part_name in parts:
                parts[part_name].set_color("#333333")
                parts[part_name].set_linewidth(1.2)

        # Strip (jittered individual points)
        for i, (topo, scores) in enumerate(zip(topologies, violin_data)):
            jitter = rng.uniform(-0.08, 0.08, size=len(scores))
            ax.scatter(
                positions[i] + jitter,
                scores,
                color=cmap[topo],
                s=22,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(topologies, rotation=15, ha="right", fontsize=_FONT_TICK)
        _apply_style(
            ax, xlabel="Topology", ylabel="Final best score", title=f"Policy: {pol}"
        )
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save(fig, path)
    return path


# ===========================================================================
# 4. Summary heatmap (topology × metric)
# ===========================================================================


def plot_summary_heatmap(
    results: list,
    output_dir: str,
    filename: str = "summary_heatmap.png",
    policy_filter: Optional[str] = None,
) -> Optional[str]:
    """Heatmap of z-scored metrics with topology rows and metric columns.

    Also saves a ``summary_heatmap_<policy>.png`` for each policy separately.

    Parameters
    ----------
    results:
        List of result dicts.
    output_dir:
        Output directory.
    filename:
        Output file name for the combined figure.
    policy_filter:
        If given, only render the heatmap for this policy.

    Returns
    -------
    str or None
    """
    plt = _get_plt()
    if plt is None:
        return None

    _metric_cfg: List[Tuple[str, str, bool]] = [
        # (key_in_results,               display_label,        higher_is_better)
        ("mean_final_score", "Final Score", True),
        ("convergence_auc_mean", "AUC", True),
        ("convergence_improvement_rate_mean", "Impr. Rate", True),
        ("convergence_plateau_fraction_mean", "Plateau Frac.", False),
        ("convergence_time_to_90pct_mean", "T→90%", False),
        ("diversity_mean_pairwise_dist", "Diversity", True),
        ("info_mean_adoption_fraction", "Adoption Frac.", True),
        ("info_num_spread_events", "Spread Events", False),
    ]

    topologies = _topology_order(results)
    policies = _policy_order(results)

    if policy_filter is not None:
        policies = [p for p in policies if p == policy_filter]

    saved_paths = []

    for pol in policies:
        pol_res = [r for r in results if r["policy"] == pol]
        # Build matrix  (n_topologies × n_metrics)
        topo_order = [r["topology"] for r in pol_res]
        matrix = np.full((len(topo_order), len(_metric_cfg)), np.nan)

        for row_i, res in enumerate(pol_res):
            for col_j, (key, _, _) in enumerate(_metric_cfg):
                val = res.get(key)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    matrix[row_i, col_j] = float(val)

        # Z-score each column (metric) independently; flip sign for "lower is better"
        z_matrix = np.full_like(matrix, np.nan)
        for col_j, (_, _, higher_better) in enumerate(_metric_cfg):
            col = matrix[:, col_j]
            valid = col[~np.isnan(col)]
            if len(valid) < 2:
                z_matrix[:, col_j] = 0.0
                continue
            mu, sigma = valid.mean(), valid.std()
            z_col = (col - mu) / (sigma + 1e-12)
            if not higher_better:
                z_col = -z_col
            z_matrix[:, col_j] = z_col

        metric_labels = [lbl for _, lbl, _ in _metric_cfg]

        # --- Figure ---
        fig, ax = plt.subplots(
            figsize=(max(10, 1.4 * len(_metric_cfg)), max(4, 0.9 * len(topo_order)))
        )
        im = ax.imshow(z_matrix, cmap="RdYlGn", aspect="auto", vmin=-2.5, vmax=2.5)

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Z-score (↑ = relatively better)", fontsize=_FONT_LABEL)
        cbar.ax.tick_params(labelsize=_FONT_TICK)

        # Labels
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=_FONT_TICK)
        ax.set_yticks(np.arange(len(topo_order)))
        ax.set_yticklabels(topo_order, fontsize=_FONT_TICK)
        ax.set_title(
            f"Topology × Metric Summary Heatmap — {pol}",
            fontsize=_FONT_TITLE,
            fontweight="bold",
            pad=10,
        )

        # Cell annotations: raw value
        for row_i in range(len(topo_order)):
            for col_j in range(len(_metric_cfg)):
                raw = matrix[row_i, col_j]
                z = z_matrix[row_i, col_j]
                if np.isnan(raw):
                    txt = "N/A"
                elif abs(raw) >= 100:
                    txt = f"{raw:.0f}"
                elif abs(raw) >= 1:
                    txt = f"{raw:.2f}"
                else:
                    txt = f"{raw:.3f}"
                txt_color = "white" if abs(z) > 1.5 else "black"
                ax.text(
                    col_j,
                    row_i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=_FONT_TICK - 1,
                    color=txt_color,
                    fontweight="bold",
                )

        fig.tight_layout()
        fname = f"summary_heatmap_{pol.lower()}.png" if len(policies) > 1 else filename
        path = os.path.join(output_dir, fname)
        _save(fig, path)
        saved_paths.append(path)

    # If more than one policy, also produce a combined figure (side by side)
    if len(policies) > 1 and not policy_filter:
        path = os.path.join(output_dir, filename)
        # Just reuse the individual files; combined is the first one saved
        print(f"  (per-policy heatmaps saved as summary_heatmap_<policy>.png)")
        return saved_paths[0] if saved_paths else None

    return saved_paths[0] if saved_paths else None


# ===========================================================================
# 5. Radar / spider chart (optional extra)
# ===========================================================================


def plot_radar_chart(
    results: list,
    output_dir: str,
    filename: str = "radar_chart.png",
    policy: str = "Trained",
) -> Optional[str]:
    """Spider / radar chart comparing topologies across normalised metrics.

    Parameters
    ----------
    results:
        List of result dicts.
    output_dir:
        Output directory.
    filename:
        Output file name.
    policy:
        Which policy's results to show.

    Returns
    -------
    str or None
    """
    plt = _get_plt()
    if plt is None:
        return None

    _axes_cfg = [
        ("mean_final_score", "Final Score", True),
        ("convergence_auc_mean", "AUC", True),
        ("diversity_mean_pairwise_dist", "Diversity", True),
        ("info_mean_adoption_fraction", "Adoption", True),
        ("info_num_spread_events", "Spread Events", False),
        ("convergence_plateau_fraction_mean", "Low Plateau", False),
    ]

    pol_res = [r for r in results if r["policy"] == policy]
    topologies = [r["topology"] for r in pol_res]
    cmap = _color_map(topologies)
    n_axes = len(_axes_cfg)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise each metric to [0, 1]
    raw: Dict[str, List[float]] = {}
    for key, _, higher_better in _axes_cfg:
        vals = [res.get(key) for res in pol_res]
        vals = [v if (v is not None and not np.isnan(v)) else 0.0 for v in vals]
        arr = np.array(vals, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            norm = np.full_like(arr, 0.5)
        else:
            norm = (arr - lo) / (hi - lo)
            if not higher_better:
                norm = 1.0 - norm
        raw[key] = norm.tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    labels = [lbl for _, lbl, _ in _axes_cfg]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=_FONT_TICK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, alpha=0.6)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="#cccccc", linewidth=0.6)
    ax.set_title(
        f"Topology Radar Chart — {policy}",
        fontsize=_FONT_TITLE,
        fontweight="bold",
        pad=20,
    )

    for i, (topo, res) in enumerate(zip(topologies, pol_res)):
        values = [raw[key][i] for key, _, _ in _axes_cfg]
        values += values[:1]
        ax.plot(angles, values, color=cmap[topo], linewidth=2, label=topo)
        ax.fill(angles, values, color=cmap[topo], alpha=0.1)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=_FONT_LEGEND,
        framealpha=0.85,
    )

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save(fig, path)
    return path


# ===========================================================================
# 6. Convenience class & plot_all()
# ===========================================================================


class ComparisonPlotter:
    """Convenience wrapper that runs all comparison plots in one call.

    Parameters
    ----------
    results:
        List of result dicts from ``eval_multi_topology.main()``.
    output_dir:
        Directory to write all figures into (created if necessary).
    """

    def __init__(self, results: list, output_dir: str) -> None:
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def convergence_curves(self, **kwargs) -> Optional[str]:
        return plot_convergence_curves(self.results, self.output_dir, **kwargs)

    def diversity_over_time(self, **kwargs) -> Optional[str]:
        return plot_diversity_over_time(self.results, self.output_dir, **kwargs)

    def fitness_distributions(self, **kwargs) -> Optional[str]:
        return plot_fitness_distributions(self.results, self.output_dir, **kwargs)

    def summary_heatmap(self, **kwargs) -> Optional[str]:
        return plot_summary_heatmap(self.results, self.output_dir, **kwargs)

    def radar_chart(self, **kwargs) -> Optional[str]:
        return plot_radar_chart(self.results, self.output_dir, **kwargs)

    def plot_all(self) -> List[str]:
        """Run all four (+ radar) plots and return list of saved paths."""
        saved = []
        for fn in [
            self.convergence_curves,
            self.diversity_over_time,
            self.fitness_distributions,
            self.summary_heatmap,
            self.radar_chart,
        ]:
            try:
                path = fn()
                if path:
                    saved.append(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{fn.__name__} failed: {exc}", RuntimeWarning)
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point — read from saved episode CSV + JSON
# ---------------------------------------------------------------------------


def _load_results_from_json(json_path: str) -> list:
    """Load the ``all_results`` list from a saved ``multi_topology_metrics.json``."""
    import json

    with open(json_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Generate comparison plots from a saved multi_topology_metrics.json"
    )
    parser.add_argument("json_path", help="Path to multi_topology_metrics.json")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output dir (default: same directory as json_path)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.json_path)
    results = _load_results_from_json(args.json_path)
    plotter = ComparisonPlotter(results, out_dir)
    paths = plotter.plot_all()
    print(f"\nGenerated {len(paths)} plots in {out_dir}")
