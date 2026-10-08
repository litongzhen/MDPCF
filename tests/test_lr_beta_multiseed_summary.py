import pytest

from scripts.summarize_lr_beta_multiseed import aggregate_rows


def test_aggregate_rows_reports_seed_count_mean_and_sample_std():
    rows = [
        {"config_id": "A", "seed": 42, "diversity": 0.4, "coverage": 1.0},
        {"config_id": "A", "seed": 43, "diversity": 0.5, "coverage": 0.9},
        {"config_id": "A", "seed": 44, "diversity": 0.6, "coverage": 0.8},
        {"config_id": "B", "seed": 42, "diversity": 0.7, "coverage": 1.0},
        {"config_id": "B", "seed": 43, "diversity": 0.8, "coverage": 1.0},
        {"config_id": "B", "seed": 44, "diversity": 0.9, "coverage": 1.0},
    ]

    summary = aggregate_rows(rows, ["diversity", "coverage"])

    a = next(row for row in summary if row["config_id"] == "A")
    assert a["seed_count"] == 3
    assert a["diversity_mean"] == pytest.approx(0.5)
    assert a["diversity_std"] == pytest.approx(0.1)
    assert a["coverage_mean"] == pytest.approx(0.9)
