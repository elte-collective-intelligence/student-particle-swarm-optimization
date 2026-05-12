import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from models import DimAgnosticNet, DimAgnosticCritic
from training.curriculum import CurriculumManager, Stage


class TestDimAgnosticNet:
    """DimAgnosticNet must accept any landscape dimension without modification."""

    @pytest.mark.parametrize("dim", [2, 5, 10, 30])
    def test_output_shapes(self, dim):
        """Output tensors have the right shape for arbitrary dim."""
        net = DimAgnosticNet(hidden_size=32)
        B, A = 4, 5
        avg_pos = torch.randn(B, A, dim)
        avg_vel = torch.randn(B, A, dim)

        outputs = net(avg_pos, avg_vel)
        assert len(outputs) == 6, "Expected 6 outputs (loc+scale for inertia/cog/soc)"
        for t in outputs:
            assert t.shape == (B, A, dim), f"Expected ({B}, {A}, {dim}), got {t.shape}"

    def test_action_ranges(self):
        """Inertia loc ≈ 0.7, cog/soc loc ≈ 1.5 when network outputs are near zero."""
        net = DimAgnosticNet(hidden_size=32)
        # zero-init decoder bias → outputs near 0 → transformations dominate
        torch.nn.init.zeros_(net.decoder.weight)
        torch.nn.init.zeros_(net.decoder.bias)

        avg_pos = torch.zeros(2, 3, 4)
        avg_vel = torch.zeros(2, 3, 4)
        inertia_loc, _, cog_loc, _, soc_loc, _ = net(avg_pos, avg_vel)

        assert torch.allclose(inertia_loc, torch.full_like(inertia_loc, 0.7), atol=1e-5)
        assert torch.allclose(cog_loc, torch.full_like(cog_loc, 1.5), atol=1e-5)
        assert torch.allclose(soc_loc, torch.full_like(soc_loc, 1.5), atol=1e-5)

    def test_critic_output_shape(self):
        critic = DimAgnosticCritic(hidden_size=32)
        B, A, D = 4, 5, 3
        val = critic(torch.randn(B, A, D), torch.randn(B, A, D))
        assert val.shape == (B, A, 1)


class TestCurriculumManager:
    """CurriculumManager must advance stages correctly."""

    def test_advances_on_threshold(self):
        """Stage should advance once rolling mean exceeds threshold."""
        mgr = CurriculumManager(
            [
                Stage("easy", 2, "sphere", max_iters=100, threshold=1.0),
                Stage("hard", 5, "sphere", max_iters=100, threshold=float("-inf")),
            ],
            window=3,
        )
        assert mgr.current.name == "easy"

        # Feed rewards below threshold — should not advance
        for _ in range(5):
            mgr.update(-1.0)
        assert mgr.current.name == "easy"

        # Feed rewards above threshold — should advance once window fills
        advanced_any = False
        for _ in range(3):
            if mgr.update(2.0):
                advanced_any = True
                break
        assert advanced_any
        assert mgr.current.name == "hard"

    def test_advances_on_max_iters(self):
        """Stage should advance on step budget even without hitting threshold."""
        mgr = CurriculumManager(
            [
                Stage("only", 2, "sphere", max_iters=5, threshold=float("inf")),
            ],
            window=10,
        )
        for _ in range(4):
            assert not mgr.done
            mgr.update(0.0)
        mgr.update(0.0)  # 5th iter
        assert mgr.done

    def test_negative_infinity_threshold_disables_early_advance(self):
        """threshold=-inf must only advance by max_iters, not by rolling mean."""
        mgr = CurriculumManager(
            [
                Stage("final", 2, "sphere", max_iters=5, threshold=float("-inf")),
            ],
            window=3,
        )
        for _ in range(4):
            assert not mgr.done
            # Any value would satisfy reward >= -inf, so this catches accidental early advance.
            mgr.update(999.0)
        mgr.update(999.0)
        assert mgr.done

    def test_preset_curricula_have_stages(self):
        """All three presets should have >= 3 stages."""
        for preset in [CurriculumManager.dimension, CurriculumManager.function, CurriculumManager.dynamics]:
            mgr = preset()
            assert len(mgr.stages) >= 3, f"{preset.__name__} has < 3 stages"

    def test_stage_log_recorded(self):
        """stage_log should be populated after advancing."""
        mgr = CurriculumManager(
            [Stage("a", 2, "sphere", max_iters=3, threshold=float("-inf"))],
            window=5,
        )
        for _ in range(3):
            mgr.update(0.0)
        assert len(mgr.stage_log) == 1
        assert mgr.stage_log[0]["stage"] == "a"
        assert mgr.stage_log[0]["iters_used"] == 3
