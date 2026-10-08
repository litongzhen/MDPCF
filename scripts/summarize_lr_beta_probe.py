import argparse
import csv
import json
from pathlib import Path


CONFIG_IDS = ("A", "B", "C", "D")


def _as_float(row, key):
    return float(row[key])


def summarize_history(history_path, config_id):
    with Path(history_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training history: {history_path}")

    best = min(rows, key=lambda row: _as_float(row, "val_objective"))
    learning_rates = [_as_float(row, "learning_rate") for row in rows]
    return {
        "config_id": config_id,
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]) + 1,
        "best_val_objective": _as_float(best, "val_objective"),
        "best_val_mse": _as_float(best, "val_mse"),
        "best_val_ce": _as_float(best, "val_ce"),
        "best_val_kl_z": _as_float(best, "val_kl_z"),
        "best_val_kl_w": _as_float(best, "val_kl_w"),
        "best_val_y": _as_float(best, "val_y"),
        "best_val_acc": _as_float(best, "val_acc"),
        "lr_changes": sum(a != b for a, b in zip(learning_rates, learning_rates[1:])),
        "final_lr": learning_rates[-1],
        "final_beta": _as_float(rows[-1], "beta"),
    }


def summarize_evaluation(evaluation_path, config_id):
    with Path(evaluation_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one evaluation row: {evaluation_path}")
    row = rows[0]
    float_fields = [
        "coverage", "validity_generated", "mean_target_prob", "diversity",
        "proximity", "lof_log_scale",
    ]
    int_fields = ["attempted_count", "generated_cf_count"]
    result = {"config_id": config_id}
    result.update({field: float(row[field]) for field in float_fields})
    result.update({field: int(float(row[field])) for field in int_fields})
    return result


def select_probe_candidates(rows):
    reference = next(row for row in rows if row["config_id"] == "A")
    minimum_accuracy = float(reference["best_val_acc"]) - 0.02
    eligible = [
        row
        for row in rows
        if row["config_id"] != "A"
        and float(row["best_val_acc"]) >= minimum_accuracy
        and float(row.get("best_val_kl_z", 1.0)) + float(row.get("best_val_kl_w", 0.0)) > 0
    ]
    eligible.sort(key=lambda row: float(row["best_val_objective"]))
    return [row["config_id"] for row in eligible[:2]]


def find_checkpoint(checkpoint_root, config_id):
    marker = f"run[lr_beta_probe_{config_id.lower()}]"
    matches = [path for path in Path(checkpoint_root).iterdir() if path.is_dir() and marker in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {config_id}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="mdpcf/csvae/ckpt/adult")
    parser.add_argument("--output-root", default="results/lr_beta_probe")
    parser.add_argument("--evaluation-root")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config_id in CONFIG_IDS:
        checkpoint = find_checkpoint(args.checkpoint_root, config_id)
        summary = summarize_history(checkpoint / "training_history.csv", config_id)
        with (checkpoint / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        summary.update(
            {
                "run_tag": config["run_tag"],
                "seed": config["seed"],
                "black_box_seed": config["black_box_seed"],
                "beta_schedule": config["beta_schedule"],
                "lr_scheduler_patience": config["lr_scheduler_patience"],
                "lr_scheduler_factor": config["lr_scheduler_factor"],
                "min_lr": config["min_lr"],
            }
        )
        summaries.append(summary)

    fieldnames = list(summaries[0])
    with (output_root / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    selected = select_probe_candidates(summaries)
    with (output_root / "selected_configs.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected_configs": selected}, handle, indent=2)
    print("Selected configurations:", ", ".join(selected))

    if args.evaluation_root:
        evaluations = []
        for config_id in ("A", *selected):
            run_tag = f"lr_beta_probe_{config_id.lower()}"
            matches = []
            for config_path in Path(args.evaluation_root).rglob("config.json"):
                with config_path.open(encoding="utf-8") as handle:
                    config = json.load(handle)
                evaluation_path = config_path.parent / "evaluation_summary.csv"
                if config.get("run_tag") == run_tag and evaluation_path.exists():
                    matches.append(evaluation_path)
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one evaluation for {config_id}, found {len(matches)}"
                )
            evaluations.append(summarize_evaluation(matches[0], config_id))
        with (output_root / "cf_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluations[0]))
            writer.writeheader()
            writer.writerows(evaluations)


if __name__ == "__main__":
    main()
