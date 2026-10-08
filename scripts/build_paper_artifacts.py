import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_failure_analysis import infer_experiment_label


MAIN_LABELS = ["mdpcf_full", "dice", "tabcf", "wachter", "revise", "cchvae"]
ABLATION_LABELS = [
    "mdpcf_full",
    "mdpcf_no_diversity",
    "mdpcf_no_latent_proximity",
    "mdpcf_no_posthoc_sparsity",
]


def load_labeled_summaries(summary_path):
    summaries = pd.read_csv(summary_path)
    summaries = summaries.copy()
    summaries["experiment_label"] = summaries.apply(infer_experiment_label, axis=1)
    return summaries


def select_protocol_rows(summaries, labels):
    rows = summaries[
        (summaries["attempted_count"] == 100)
        & (summaries["expected_cf_count"] == 300)
        & (summaries["experiment_label"].isin(labels))
    ].copy()
    rows["label_order"] = rows["experiment_label"].apply(labels.index)
    rows = rows.sort_values("label_order").drop(columns=["label_order"])
    return rows


def write_metric_table(rows, output_path):
    columns = [
        "dataset",
        "experiment_label",
        "coverage",
        "validity",
        "validity_generated",
        "diversity",
        "raw_diversity",
        "proximity",
        "sparsity",
        "latent_nn_distance",
        "constraint_violation_rate",
        "generated_cf_count",
        "valid_cf_count",
    ]
    table = rows[columns].copy()
    numeric_cols = [col for col in columns if col not in {"dataset", "experiment_label"}]
    table[numeric_cols] = table[numeric_cols].apply(pd.to_numeric, errors="coerce")
    table.to_csv(output_path, index=False)
    return table


def plot_metric_bars(table, metrics, output_path, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(table["experiment_label"], table[metric], color="#3b6ea8")
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build paper-ready result artifacts.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/evaluation_summaries.csv"),
    )
    parser.add_argument(
        "--failure",
        type=Path,
        default=Path("results/failure_analysis_summary.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_labeled_summaries(args.summary)
    main_rows = select_protocol_rows(summaries, MAIN_LABELS)
    ablation_rows = select_protocol_rows(summaries, ABLATION_LABELS)

    main_table = write_metric_table(main_rows, args.output_dir / "main_results_100.csv")
    ablation_table = write_metric_table(
        ablation_rows, args.output_dir / "ablation_results_100.csv"
    )

    failures = pd.read_csv(args.failure)
    failure_rows = failures[
        (failures["attempted_count"] == 100)
        & (failures["expected_cf_count"] == 300)
        & (failures["experiment_label"].isin(MAIN_LABELS + ABLATION_LABELS))
    ].copy()
    failure_rows.to_csv(args.output_dir / "failure_analysis_100.csv", index=False)

    plot_metric_bars(
        main_table,
        ["coverage", "validity", "diversity", "proximity"],
        figures_dir / "main_results_100.png",
        "Adult main results, 100 fixed test instances",
    )
    plot_metric_bars(
        ablation_table,
        ["coverage", "validity", "diversity", "sparsity", "latent_nn_distance"],
        figures_dir / "ablation_results_100.png",
        "MDPCF ablation results, 100 fixed test instances",
    )

    print(f"Wrote paper artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
