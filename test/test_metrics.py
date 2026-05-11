import math
import sys
from pathlib import Path

# Add src to path before importing project modules
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from eval.convergence import aggregate_convergence_metrics  # noqa: E402
from eval.convergence import compute_episode_convergence  # noqa: E402
from eval.diversity import DiversityTracker  # noqa: E402
from eval.diversity import compute_diversity_metrics  # noqa: E402
from eval.diversity import mean_pairwise_distance  # noqa: E402
from eval.diversity import position_spread  # noqa: E402
from eval.diversity import velocity_alignment  # noqa: E402
from eval.diversity import velocity_speed_std  # noqa: E402
from eval.information_spread import InformationSpreadTracker  # noqa: E402
from eval.information_spread import information_adoption_fraction  # noqa: E402
from eval.information_spread import information_entropy  # noqa: E402


class TestDiversityMetrics:
    def test_mean_pairwise_distance_matches_toy_triangle(self):
        positions = torch.tensor(
            [[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]],
            dtype=torch.float32,
        )

        result = mean_pairwise_distance(positions)

        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_position_spread_aabb_matches_rectangle_area(self):
        positions = torch.tensor(
            [[[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [3.0, 2.0]]],
            dtype=torch.float32,
        )

        result = position_spread(positions, use_hull=False)

        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([6.0]))

    def test_velocity_metrics_have_expected_ranges_on_toy_vectors(self):
        velocities = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
            dtype=torch.float32,
        )

        speed_std = velocity_speed_std(velocities)
        alignment = velocity_alignment(velocities)

        assert speed_std.shape == (1,)
        assert torch.allclose(speed_std, torch.tensor([0.0]))
        assert alignment.shape == (1,)
        assert torch.allclose(alignment, torch.tensor([-1.0 / 3.0]), atol=1e-6)
        assert torch.all(alignment >= -1.0)
        assert torch.all(alignment <= 1.0)

    def test_compute_diversity_metrics_returns_finite_batched_scalars(self):
        positions = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        velocities = torch.zeros_like(positions)

        metrics = compute_diversity_metrics(
            positions=positions,
            velocities=velocities,
            use_hull=False,
        )

        assert set(metrics) == {
            "mean_pairwise_dist",
            "position_spread",
            "velocity_speed_std",
            "velocity_alignment",
        }
        for value in metrics.values():
            assert value.shape == (2,)
            assert torch.isfinite(value).all()

        assert torch.all(metrics["mean_pairwise_dist"] >= 0.0)
        assert torch.all(metrics["position_spread"] >= 0.0)
        assert torch.all(metrics["velocity_alignment"] >= -1.0)
        assert torch.all(metrics["velocity_alignment"] <= 1.0)

    def test_zero_velocities_keep_alignment_finite(self):
        velocities = torch.zeros((2, 4, 3), dtype=torch.float32)

        alignment = velocity_alignment(velocities)

        assert alignment.shape == (2,)
        assert torch.isfinite(alignment).all()
        assert torch.allclose(alignment, torch.zeros(2))

    def test_diversity_tracker_records_stepwise_history(self):
        tracker = DiversityTracker(use_hull=False, batch_index=0)
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        velocities = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
            dtype=torch.float32,
        )

        first = tracker.update(positions, velocities)
        second = tracker.update(positions + 1.0, velocities * 0.0)
        summary = tracker.summarise()

        assert set(first) == {
            "mean_pairwise_dist",
            "position_spread",
            "velocity_speed_std",
            "velocity_alignment",
        }
        assert set(second) == set(first)
        for key, values in summary.items():
            assert len(values) == 2
            assert all(math.isfinite(v) for v in values)


class TestInformationSpreadMetrics:
    def test_information_adoption_fraction_handles_negative_scores(self):
        neighborhood_best_scores = torch.tensor(
            [[-1.0, -1.0, -2.0, -3.0], [0.0, -1e-9, -1.0, -2.0]],
            dtype=torch.float32,
        )
        global_best_score = torch.tensor([-1.0, 0.0], dtype=torch.float32)

        result = information_adoption_fraction(
            neighborhood_best_scores,
            global_best_score,
        )

        assert result.shape == (2,)
        assert torch.allclose(result, torch.tensor([0.5, 0.25]))
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_information_entropy_is_finite_at_boundaries(self):
        adoption_fraction = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

        entropy = information_entropy(adoption_fraction)

        assert entropy.shape == (3,)
        assert torch.isfinite(entropy).all()
        assert torch.allclose(entropy, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
        assert torch.all(entropy >= 0.0)
        assert torch.all(entropy <= 1.0 + 1e-6)

    def test_information_spread_tracker_summarises_deterministic_event_curves(self):
        tracker = InformationSpreadTracker(batch_index=0)

        steps = [
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            ),
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            ),
            (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            ),
            (
                torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            ),
        ]

        per_step = [tracker.update(pb, nb) for pb, nb in steps]
        summary = tracker.summarise()
        means = tracker.mean_over_episode()

        assert per_step[0]["is_new_best"] == 1.0
        assert per_step[1]["adoption_fraction"] == 0.5
        assert per_step[2]["adoption_fraction"] == 1.0
        assert per_step[3]["is_new_best"] == 1.0

        assert summary["total_steps"] == 4
        assert summary["num_spread_events"] == 2
        assert len(summary["adoption_fraction_curve"]) == 4
        assert len(summary["information_entropy_curve"]) == 4
        assert len(summary["spread_events"]) == 2
        assert summary["spread_events"][0]["steps_to_50pct"] == 1
        assert summary["spread_events"][0]["steps_to_90pct"] == 2
        assert summary["spread_events"][0]["steps_to_full"] == 2
        assert summary["spread_events"][0]["peak_adoption"] == 1.0
        assert summary["final_adoption_fraction"] == 0.25
        assert 0.0 <= means["mean_adoption_fraction"] <= 1.0
        assert 0.0 <= means["mean_information_entropy"] <= 1.0


class TestConvergenceMetrics:
    def test_compute_episode_convergence_matches_linear_toy_curve(self):
        curve = [0.0, 1.0, 2.0, 3.0]

        metrics = compute_episode_convergence(curve)

        assert metrics["auc"] == pytest.approx(0.5)
        assert metrics["time_to_50pct"] == 2
        assert metrics["time_to_80pct"] == 3
        assert metrics["time_to_90pct"] == 3
        assert metrics["time_to_99pct"] == 3
        assert metrics["final_score"] == pytest.approx(3.0)
        assert metrics["improvement_rate"] == pytest.approx(1.0)
        assert metrics["plateau_fraction"] == pytest.approx(0.0)

    def test_compute_episode_convergence_handles_empty_curve(self):
        metrics = compute_episode_convergence([])

        assert metrics["auc"] == 0.0
        assert metrics["time_to_50pct"] is None
        assert metrics["time_to_80pct"] is None
        assert metrics["time_to_90pct"] is None
        assert metrics["time_to_99pct"] is None
        assert math.isnan(metrics["final_score"])
        assert metrics["improvement_rate"] == 0.0
        assert metrics["plateau_fraction"] == 0.0

    def test_aggregate_convergence_metrics_pads_curves_and_ignores_none_times(self):
        curves = [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0],
        ]

        metrics = aggregate_convergence_metrics(curves)

        assert metrics["auc_mean"] == pytest.approx(0.25)
        assert metrics["time_to_50pct_mean"] == 2.0
        assert metrics["time_to_80pct_mean"] == 3.0
        assert metrics["time_to_90pct_mean"] == 3.0
        assert metrics["time_to_99pct_mean"] == 3.0
        assert metrics["final_score_mean"] == pytest.approx(1.5)
        assert metrics["improvement_rate_mean"] == pytest.approx(0.5)
        assert metrics["plateau_fraction_mean"] == pytest.approx(0.5)
        assert metrics["mean_curve"] == pytest.approx([0.0, 0.5, 1.0, 1.5])
        assert len(metrics["std_curve"]) == 4
        assert all(math.isfinite(v) for v in metrics["std_curve"])
