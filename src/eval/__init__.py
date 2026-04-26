"""eval sub-package — diversity and analysis utilities for PSO."""

from eval.diversity import (
    DiversityTracker,
    compute_diversity_metrics,
    mean_pairwise_distance,
    position_spread,
    velocity_alignment,
    velocity_speed_std,
)
from eval.information_spread import (
    InformationSpreadTracker,
    information_adoption_fraction,
    information_entropy,
)
from eval.convergence import (
    compute_episode_convergence,
    aggregate_convergence_metrics,
)
from eval.comparison_plots import (
    ComparisonPlotter,
    plot_convergence_curves,
    plot_diversity_over_time,
    plot_fitness_distributions,
    plot_summary_heatmap,
    plot_radar_chart,
)

__all__ = [
    # diversity
    "DiversityTracker",
    "compute_diversity_metrics",
    "mean_pairwise_distance",
    "position_spread",
    "velocity_alignment",
    "velocity_speed_std",
    # information spread
    "InformationSpreadTracker",
    "information_adoption_fraction",
    "information_entropy",
    # convergence
    "compute_episode_convergence",
    "aggregate_convergence_metrics",
    # comparison plots
    "ComparisonPlotter",
    "plot_convergence_curves",
    "plot_diversity_over_time",
    "plot_fitness_distributions",
    "plot_summary_heatmap",
    "plot_radar_chart",
]

