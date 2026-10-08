import csv
import math
from pathlib import Path

try:
    from scripts.summarize_cruds_csvae_probe import (
        METRICS,
        find_evaluation,
        parse_metric,
        reference_root,
    )
except ModuleNotFoundError:
    from summarize_cruds_csvae_probe import (
        METRICS,
        find_evaluation,
        parse_metric,
        reference_root,
    )


CURRENT_RUN_TAG = "cruds_tokenw_w4_wkl1_e300"
CURRENT_ROOTS = {
    "adult": "results/cruds_tokenw_w4_plaus005_adult_n20",
    "bank": "results/cruds_tokenw_w4_wkl1_e300_n20",
    "lending-club": "results/cruds_tokenw_w4_plaus005_lending_n20",
}


def build_rows():
    rows = []
    for dataset, current_root in CURRENT_ROOTS.items():
        current = find_evaluation(current_root, dataset, CURRENT_RUN_TAG)
        reference = find_evaluation(
            reference_root(dataset), dataset, "lr_beta_probe_a"
        )
        row = {"dataset": dataset}
        for metric in METRICS:
            new_value = parse_metric(current, metric)
            old_value = parse_metric(reference, metric)
            row[f"reference_{metric}"] = old_value
            row[f"cruds_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value
        tradeoff_available = math.isfinite(row["delta_diversity"]) and math.isfinite(
            row["delta_lof_log_scale"]
        )
        row["coverage_gate"] = row["delta_coverage"] >= -0.02
        row["validity_gate"] = row["delta_validity_generated"] >= -0.03
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
    output = Path("results/cruds_tokenw_final_probe")
    output.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CRUDS-aligned token-w final probe",
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
            "Adult and Lending Club use latent GMM weight 0.05; Bank uses 0.0, matching the reference generation configurations.",
        ]
    )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
