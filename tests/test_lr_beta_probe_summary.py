import csv
from pathlib import Path

from scripts.summarize_lr_beta_probe import (
    select_probe_candidates,
    summarize_evaluation,
    summarize_history,
)


def _write_history(path: Path, objectives, accuracies=None):
    accuracies = accuracies or [0.9] * len(objectives)
    fieldnames = [
        "epoch", "beta", "learning_rate", "train_mse", "train_ce", "train_kl",
        "val_mse", "val_ce", "val_kl_z", "val_kl_w", "val_y", "val_acc",
        "val_objective", "is_best",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, (objective, accuracy) in enumerate(zip(objectives, accuracies)):
            writer.writerow(
                {
                    "epoch": epoch,
                    "beta": 0.001,
                    "learning_rate": 0.001 if epoch < 2 else 0.0005,
                    "train_mse": 0.1,
                    "train_ce": 0.2,
                    "train_kl": 0.3,
                    "val_mse": objective / 10,
                    "val_ce": objective / 5,
                    "val_kl_z": 0.4,
                    "val_kl_w": 0.1,
                    "val_y": objective * 0.7,
                    "val_acc": accuracy,
                    "val_objective": objective,
                    "is_best": int(objective == min(objectives[: epoch + 1])),
                }
            )


def test_summarize_history_reports_best_row_and_schedule_changes(tmp_path):
    history = tmp_path / "training_history.csv"
    _write_history(history, [0.5, 0.4, 0.45])

    summary = summarize_history(history, "B")

    assert summary["config_id"] == "B"
    assert summary["best_epoch"] == 2
    assert summary["best_val_objective"] == 0.4
    assert summary["lr_changes"] == 1
    assert summary["final_beta"] == 0.001


def test_candidate_selection_uses_validation_and_rejects_accuracy_collapse():
    rows = [
        {"config_id": "A", "best_val_objective": 0.50, "best_val_acc": 0.90},
        {"config_id": "B", "best_val_objective": 0.40, "best_val_acc": 0.91},
        {"config_id": "C", "best_val_objective": 0.30, "best_val_acc": 0.70},
        {"config_id": "D", "best_val_objective": 0.45, "best_val_acc": 0.90},
    ]

    assert select_probe_candidates(rows) == ["B", "D"]


def test_summarize_evaluation_extracts_counterfactual_metrics(tmp_path):
    path = tmp_path / "evaluation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "coverage", "validity_generated", "mean_target_prob", "diversity",
                "proximity", "lof_log_scale", "attempted_count", "generated_cf_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "coverage": 0.95,
                "validity_generated": 1.0,
                "mean_target_prob": 0.7,
                "diversity": 0.4,
                "proximity": 0.3,
                "lof_log_scale": 2.0,
                "attempted_count": 20,
                "generated_cf_count": 180,
            }
        )

    result = summarize_evaluation(path, "B")

    assert result["config_id"] == "B"
    assert result["coverage"] == 0.95
    assert result["generated_cf_count"] == 180
