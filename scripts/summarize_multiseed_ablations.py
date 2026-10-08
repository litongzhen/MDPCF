"""Aggregate seed-specific MDPCE ablation summaries into paper-ready CSV files."""

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "coverage",
    "validity_generated",
    "mean_target_prob",
    "diversity",
    "proximity",
    "lof_log_scale",
)


def variant_from_config(config):
    if config.get("latent_model", "csvae") == "standard_vae":
        return "standard_vae"
    if config.get("csvae_backbone", "transformer") == "mlp":
        return "csvae_mlp"
    return "csvae_transformer"


def build_summary_tables(results_root):
    """Load all complete evaluation summaries below *results_root*."""
    records = []
    for summary_path in sorted(Path(results_root).rglob("evaluation_summary.csv")):
        config_path = summary_path.with_name("config.json")
        if not config_path.is_file():
            continue

        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        if not config.get("run_tag"):
            continue

        summary = pd.read_csv(summary_path)
        if len(summary) != 1 or any(metric not in summary.columns for metric in METRICS):
            continue

        row = {
            "dataset": config["dataname"],
            "variant": variant_from_config(config),
            "seed": int(config["seed"]),
            # Legacy runs predate the fixed-black-box protocol, so their own
            # seed identifies the black-box artifact they used.
            "black_box_seed": int(config.get("black_box_seed", config["seed"])),
            "summary_path": str(summary_path),
        }
        row.update(summary.iloc[0][list(METRICS)].to_dict())
        records.append(row)

    per_seed = pd.DataFrame(records)
    if per_seed.empty:
        return per_seed, pd.DataFrame()

    per_seed = per_seed.sort_values(["dataset", "variant", "seed"]).reset_index(drop=True)
    black_box_seed_counts = per_seed.groupby(["dataset", "variant"])["black_box_seed"].nunique()
    inconsistent_groups = black_box_seed_counts[black_box_seed_counts > 1]
    if not inconsistent_groups.empty:
        groups = ", ".join(
            f"{dataset}/{variant}" for dataset, variant in inconsistent_groups.index
        )
        raise ValueError(
            "Cannot aggregate runs with different black_box_seed values: " + groups
        )
    aggregation = per_seed.groupby(["dataset", "variant"], as_index=False)[list(METRICS)].agg(["mean", "std"])
    aggregation.columns = [
        "_".join(filter(None, column)).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregation.columns
    ]
    counts = per_seed.groupby(["dataset", "variant"], as_index=False).size().rename(columns={"size": "n_seeds"})
    aggregate = counts.merge(aggregation, on=["dataset", "variant"], how="inner")
    return per_seed, aggregate.sort_values(["dataset", "variant"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/multiseed_ablation/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multiseed_ablation"))
    args = parser.parse_args()

    per_seed, aggregate = build_summary_tables(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    aggregate.to_csv(args.output_dir / "mean_std.csv", index=False)
    print(f"Wrote {len(per_seed)} per-seed rows and {len(aggregate)} aggregate rows to {args.output_dir}")


if __name__ == "__main__":
    main()
