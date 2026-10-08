import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.summarize_lr_beta_probe import summarize_evaluation, summarize_history
except ModuleNotFoundError:
    from summarize_lr_beta_probe import summarize_evaluation, summarize_history


DATASETS = ("bank", "default", "gmc", "lending-club")
CONFIG_IDS = ("A", "B")


def compare_dataset_rows(rows):
    comparisons = []
    for dataset in sorted({row["dataset"] for row in rows}):
        a = next(row for row in rows if row["dataset"] == dataset and row["config_id"] == "A")
        b = next(row for row in rows if row["dataset"] == dataset and row["config_id"] == "B")
        delta = lambda field: round(float(b[field]) - float(a[field]), 12)
        diversity_delta = delta("diversity")
        lof_delta = delta("lof_log_scale")
        comparisons.append(
            {
                "dataset": dataset,
                "coverage_delta": delta("coverage"),
                "validity_delta": delta("validity_generated"),
                "target_probability_delta": delta("mean_target_prob"),
                "diversity_delta": diversity_delta,
                "proximity_delta": delta("proximity"),
                "lof_delta": lof_delta,
                "generated_cf_count_delta": int(b["generated_cf_count"]) - int(a["generated_cf_count"]),
                "coverage_non_decrease": float(b["coverage"]) >= float(a["coverage"]),
                "validity_non_decrease": float(b["validity_generated"]) >= float(a["validity_generated"]),
                "joint_diversity_lof_improvement": diversity_delta > 0 and lof_delta < 0,
            }
        )
    return comparisons


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find_training_checkpoint(checkpoint_root, dataset, config_id):
    marker = f"run[lr_beta_probe_{config_id.lower()}]_seed[42]"
    root = Path(checkpoint_root) / dataset
    matches = [path for path in root.iterdir() if path.is_dir() and marker in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {dataset}/{config_id}, found {len(matches)}")
    return matches[0]


def _find_evaluation(evaluation_root, dataset, config_id):
    run_tag = f"lr_beta_probe_{config_id.lower()}"
    matches = []
    for config_path in Path(evaluation_root).rglob("config.json"):
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        evaluation_path = config_path.parent / "evaluation_summary.csv"
        if (
            config.get("dataname") == dataset
            and config.get("run_tag") == run_tag
            and int(config.get("seed", -1)) == 42
            and int(config.get("black_box_seed", -1)) == 42
            and int(config.get("num_samples", -1)) == 20
            and evaluation_path.exists()
        ):
            matches.append(evaluation_path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one evaluation for {dataset}/{config_id}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="mdpcf/csvae/ckpt")
    parser.add_argument("--evaluation-root", default="results/lr_beta_cross_dataset/runs")
    parser.add_argument("--output-root", default="results/lr_beta_cross_dataset")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    training_rows = []
    cf_rows = []
    for dataset in DATASETS:
        for config_id in CONFIG_IDS:
            checkpoint = _find_training_checkpoint(args.checkpoint_root, dataset, config_id)
            training = summarize_history(checkpoint / "training_history.csv", config_id)
            training["dataset"] = dataset
            training_rows.append(training)
            evaluation = summarize_evaluation(
                _find_evaluation(args.evaluation_root, dataset, config_id), config_id
            )
            evaluation["dataset"] = dataset
            cf_rows.append(evaluation)

    _write_csv(output / "training_per_dataset.csv", training_rows)
    _write_csv(output / "cf_per_dataset.csv", cf_rows)
    _write_csv(output / "comparison.csv", compare_dataset_rows(cf_rows))


if __name__ == "__main__":
    main()
