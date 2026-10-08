import argparse
import csv
import json
import statistics
from pathlib import Path

try:
    from scripts.summarize_lr_beta_probe import summarize_evaluation, summarize_history
except ModuleNotFoundError:
    from summarize_lr_beta_probe import summarize_evaluation, summarize_history


CONFIG_IDS = ("A", "B")
SEEDS = (42, 43, 44)


def aggregate_rows(rows, metric_fields):
    summaries = []
    for config_id in sorted({row["config_id"] for row in rows}):
        group = [row for row in rows if row["config_id"] == config_id]
        summary = {"config_id": config_id, "seed_count": len(group)}
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            summary[f"{field}_mean"] = statistics.mean(values)
            summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find_training_checkpoint(root, config_id, seed):
    marker = f"run[lr_beta_probe_{config_id.lower()}]_seed[{seed}]"
    matches = [path for path in Path(root).iterdir() if path.is_dir() and marker in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {config_id}/seed {seed}, found {len(matches)}")
    return matches[0]


def _find_evaluation(root, config_id, seed):
    run_tag = f"lr_beta_probe_{config_id.lower()}"
    matches = []
    for config_path in Path(root).rglob("config.json"):
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        evaluation_path = config_path.parent / "evaluation_summary.csv"
        if (
            config.get("run_tag") == run_tag
            and int(config.get("seed", -1)) == seed
            and int(config.get("black_box_seed", -1)) == 42
            and evaluation_path.exists()
        ):
            matches.append(evaluation_path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one evaluation for {config_id}/seed {seed}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="mdpcf/csvae/ckpt/adult")
    parser.add_argument("--evaluation-root", default="results/lr_beta_probe/runs")
    parser.add_argument("--output-root", default="results/lr_beta_probe")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    training_rows = []
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            checkpoint = _find_training_checkpoint(args.checkpoint_root, config_id, seed)
            row = summarize_history(checkpoint / "training_history.csv", config_id)
            row["seed"] = seed
            training_rows.append(row)
    _write_csv(output / "multiseed_training_per_seed.csv", training_rows)
    training_metrics = [
        "best_val_objective", "best_val_mse", "best_val_ce", "best_val_kl_z",
        "best_val_kl_w", "best_val_y", "best_val_acc",
    ]
    _write_csv(
        output / "multiseed_training_mean_std.csv",
        aggregate_rows(training_rows, training_metrics),
    )

    cf_rows = []
    for config_id in CONFIG_IDS:
        for seed in SEEDS:
            row = summarize_evaluation(
                _find_evaluation(args.evaluation_root, config_id, seed), config_id
            )
            row["seed"] = seed
            cf_rows.append(row)
    _write_csv(output / "multiseed_cf_per_seed.csv", cf_rows)
    cf_metrics = [
        "coverage", "validity_generated", "mean_target_prob", "diversity",
        "proximity", "lof_log_scale", "generated_cf_count",
    ]
    _write_csv(
        output / "multiseed_cf_mean_std.csv",
        aggregate_rows(cf_rows, cf_metrics),
    )


if __name__ == "__main__":
    main()
