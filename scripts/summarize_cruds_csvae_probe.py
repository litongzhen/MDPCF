import argparse
import csv
import json
import math
from pathlib import Path


DATASETS = ("adult", "bank", "lending-club")
METRICS = (
    "coverage",
    "validity_generated",
    "mean_target_prob",
    "diversity",
    "proximity",
    "lof_log_scale",
)


def find_evaluation(root, dataset, run_tag):
    matches = []
    for config_path in Path(root).rglob("config.json"):
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        evaluation = config_path.parent / "evaluation_summary.csv"
        if (
            config.get("dataname") == dataset
            and config.get("run_tag") == run_tag
            and int(config.get("seed", -1)) == 42
            and int(config.get("black_box_seed", -1)) == 42
            and int(config.get("num_samples", -1)) == 20
            and config.get("latent_model", "csvae") == "csvae"
            and config.get("csvae_backbone", "transformer") == "transformer"
            and evaluation.is_file()
        ):
            matches.append(evaluation)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one evaluation for {dataset}/{run_tag}, found {len(matches)}"
        )
    with matches[0].open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def reference_root(dataset):
    if dataset == "adult":
        return "results/lr_beta_probe/runs"
    return "results/lr_beta_cross_dataset/runs"


def parse_metric(row, metric):
    value = row.get(metric, "")
    return float(value) if value not in {"", None} else math.nan


def build_rows(current_root, current_run_tag):
    rows = []
    for dataset in DATASETS:
        current = find_evaluation(
            current_root, dataset, current_run_tag
        )
        reference = find_evaluation(reference_root(dataset), dataset, "lr_beta_probe_a")
        row = {"dataset": dataset}
        for metric in METRICS:
            new_value = parse_metric(current, metric)
            old_value = parse_metric(reference, metric)
            row[f"reference_{metric}"] = old_value
            row[f"cruds_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value
        row["coverage_gate"] = row["delta_coverage"] >= -0.02
        row["validity_gate"] = row["delta_validity_generated"] >= -0.03
        tradeoff_available = math.isfinite(row["delta_diversity"]) and math.isfinite(
            row["delta_lof_log_scale"]
        )
        row["diversity_lof_gate"] = tradeoff_available and not (
            row["delta_diversity"] < 0 and row["delta_lof_log_scale"] > 0
        )
        row["dataset_gate"] = (
            row["coverage_gate"]
            and row["validity_gate"]
            and row["diversity_lof_gate"]
        )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-root", default="results/cruds_globalw_e1000_n20_nolatent"
    )
    parser.add_argument("--current-run-tag", default="cruds_globalw_e1000")
    parser.add_argument("--output-root", default="results/cruds_globalw_probe")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.current_root, args.current_run_tag)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CRUDS-aligned CSVAE probe",
        "",
        "| Dataset | Coverage delta | Validity delta | Diversity delta | LOF delta | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['delta_coverage']:.4f} | "
            f"{row['delta_validity_generated']:.4f} | {row['delta_diversity']:.4f} | "
            f"{row['delta_lof_log_scale']:.4f} | {'PASS' if row['dataset_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Overall gate: {'PASS' if all(row['dataset_gate'] for row in rows) else 'FAIL'}.",
            "",
            "Positive diversity deltas are better; negative LOF deltas are better.",
        ]
    )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
