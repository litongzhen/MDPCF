from scripts.summarize_lr_beta_cross_dataset import compare_dataset_rows


def test_compare_dataset_rows_marks_joint_diversity_lof_improvement():
    rows = [
        {
            "dataset": "bank", "config_id": "A", "coverage": 1.0,
            "validity_generated": 1.0, "mean_target_prob": 0.55,
            "diversity": 0.4, "proximity": 0.3, "lof_log_scale": 2.5,
            "generated_cf_count": 180,
        },
        {
            "dataset": "bank", "config_id": "B", "coverage": 1.0,
            "validity_generated": 1.0, "mean_target_prob": 0.56,
            "diversity": 0.45, "proximity": 0.31, "lof_log_scale": 2.3,
            "generated_cf_count": 185,
        },
    ]

    result = compare_dataset_rows(rows)[0]

    assert result["dataset"] == "bank"
    assert result["diversity_delta"] == 0.05
    assert result["lof_delta"] == -0.2
    assert result["joint_diversity_lof_improvement"] is True
    assert result["coverage_non_decrease"] is True
    assert result["validity_non_decrease"] is True
