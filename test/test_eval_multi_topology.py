import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add src to path before importing project modules
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from eval_multi_topology import (  # noqa: E402
    _aggregate_for_plots,
    _make_env,
    _prepare_model_path,
    _save_episode_csv,
)


def test_topology_sweep_config_declares_expected_grid():
    cfg = OmegaConf.load(root_dir / "src" / "configs" / "topology_sweep.yaml")

    assert [topo.name for topo in cfg.topologies] == [
        "global",
        "ring",
        "von_neumann",
        "knearest",
    ]
    assert list(cfg.landscapes) == ["sphere", "rastrigin", "dynamic_sphere"]
    assert list(cfg.seeds) == [0, 1, 2, 3, 4]
    assert cfg.env.num_agents == 10
    assert cfg.model_path == "src/outputs/full_train/best_model.pt"


def test_make_env_uses_explicit_landscape_for_sweep_configs_without_default():
    cfg = OmegaConf.create(
        {
            "env": {
                "landscape_dim": 2,
                "num_agents": 4,
                "delta": 1.0,
            }
        }
    )

    env, landscape_fn = _make_env(
        cfg,
        topology_cfg={"type": "global"},
        seed=123,
        device=torch.device("cpu"),
        landscape_name="rastrigin",
    )

    point = torch.tensor([[0.5, 0.0]])
    assert landscape_fn(point).item() == np.float32(-20.25)
    assert env.base_env.num_agents == 4
    env.close()


def test_aggregate_for_plots_groups_repeated_seeds_by_landscape():
    results = [
        {
            "landscape": "sphere",
            "seed": 0,
            "topology": "global",
            "policy": "Trained",
            "mean_final_score": 1.0,
            "convergence_auc_mean": 10.0,
            "convergence_mean_curve": [1.0, 3.0],
            "episode_histories": [{"episode": 0}],
            "convergence_per_episode": [{"final_score": 3.0}],
        },
        {
            "landscape": "sphere",
            "seed": 1,
            "topology": "global",
            "policy": "Trained",
            "mean_final_score": 3.0,
            "convergence_auc_mean": 14.0,
            "convergence_mean_curve": [3.0],
            "episode_histories": [{"episode": 1}],
            "convergence_per_episode": [{"final_score": 5.0}],
        },
        {
            "landscape": "rastrigin",
            "seed": 0,
            "topology": "global",
            "policy": "Trained",
            "mean_final_score": 99.0,
            "convergence_mean_curve": [99.0],
        },
    ]

    aggregated = _aggregate_for_plots(results, landscape="sphere")

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["landscape"] == "sphere"
    assert row["topology"] == "global"
    assert row["policy"] == "Trained"
    assert row["seeds"] == [0, 1]
    assert row["mean_final_score"] == 2.0
    assert row["convergence_auc_mean"] == 12.0
    assert row["convergence_mean_curve"] == [2.0, 3.0]
    assert row["convergence_std_curve"] == [1.0, 0.0]
    assert row["episode_histories"] == [{"episode": 0}, {"episode": 1}]
    assert row["convergence_per_episode"] == [
        {"final_score": 3.0},
        {"final_score": 5.0},
    ]


def test_save_episode_csv_includes_landscape_and_seed(tmp_path):
    results = [
        {
            "landscape": "sphere",
            "seed": 7,
            "topology": "ring",
            "policy": "Trained",
            "episode_histories": [
                {
                    "episode": 0,
                    "diversity": {"mean_pairwise_dist": [1.0, 3.0]},
                    "info_spread": {"num_spread_events": 2},
                }
            ],
            "convergence_per_episode": [
                {
                    "auc": 1.5,
                    "final_score": -0.25,
                    "plateau_fraction": 0.0,
                }
            ],
        }
    ]

    csv_path = _save_episode_csv(str(tmp_path), results)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["landscape"] == "sphere"
    assert rows[0]["seed"] == "7"
    assert rows[0]["topology"] == "ring"
    assert rows[0]["policy"] == "Trained"
    assert rows[0]["episode"] == "0"
    assert rows[0]["auc"] == "1.5"
    assert rows[0]["final_score"] == "-0.25"
    assert rows[0]["mean_pairwise_dist"] == "2.0"
    assert rows[0]["num_spread_events"] == "2"


def test_prepare_model_path_copies_checkpoint_to_timestamped_run_dir(tmp_path):
    original_model = tmp_path / "best_model.pt"
    original_model.write_bytes(b"checkpoint")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    model_path, copied = _prepare_model_path(
        str(original_model), str(output_dir), "20260508_120000"
    )

    copied_path = output_dir / "best_model_20260508_120000.pt"
    assert copied is True
    assert model_path == str(copied_path)
    assert copied_path.read_bytes() == b"checkpoint"


def test_prepare_model_path_returns_original_when_checkpoint_missing(tmp_path):
    missing_model = tmp_path / "missing.pt"
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    model_path, copied = _prepare_model_path(
        str(missing_model), str(output_dir), "20260508_120000"
    )

    assert copied is False
    assert model_path == str(missing_model)
    assert not (output_dir / "best_model_20260508_120000.pt").exists()
