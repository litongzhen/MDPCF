import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_multiseed_ablations import build_summary_tables


def _write_run(root, dataset, variant, seed, diversity, lof, black_box_seed=42):
    run_dir = root / dataset / variant / f"seed-{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"dataname": dataset, "seed": seed, "black_box_seed": black_box_seed, "run_tag": "multiseed", "latent_model": "standard_vae" if variant == "standard_vae" else "csvae", "csvae_backbone": "mlp" if variant == "csvae_mlp" else "transformer"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{
            "coverage": 1.0,
            "validity_generated": 0.9,
            "mean_target_prob": 0.8,
            "diversity": diversity,
            "proximity": 0.2,
            "lof_log_scale": lof,
        }]
    ).to_csv(run_dir / "evaluation_summary.csv", index=False)


def test_build_summary_tables_returns_sample_mean_and_std(tmp_path):
    _write_run(tmp_path, "adult", "csvae_transformer", 42, 0.2, 2.0)
    _write_run(tmp_path, "adult", "csvae_transformer", 43, 0.4, 4.0)
    _write_run(tmp_path, "adult", "csvae_transformer", 44, 0.6, 6.0)

    per_seed, aggregate = build_summary_tables(tmp_path)

    assert per_seed["seed"].tolist() == [42, 43, 44]
    row = aggregate.iloc[0]
    assert row["n_seeds"] == 3
    assert row["diversity_mean"] == pytest.approx(0.4)
    assert row["diversity_std"] == pytest.approx(0.2)
    assert row["lof_log_scale_mean"] == pytest.approx(4.0)
    assert row["lof_log_scale_std"] == pytest.approx(2.0)
    assert row["validity_generated_mean"] == pytest.approx(0.9)
    assert row["mean_target_prob_mean"] == pytest.approx(0.8)
    assert row["proximity_mean"] == pytest.approx(0.2)


def test_build_summary_tables_rejects_mixed_black_box_seeds(tmp_path):
    _write_run(tmp_path, "adult", "csvae_transformer", 42, 0.2, 2.0, black_box_seed=42)
    _write_run(tmp_path, "adult", "csvae_transformer", 43, 0.4, 4.0, black_box_seed=43)

    with pytest.raises(ValueError, match="black_box_seed"):
        build_summary_tables(tmp_path)
