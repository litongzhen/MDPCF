import argparse
from pathlib import Path

import pandas as pd


def collect_evaluation_summaries(results_root):
    results_root = Path(results_root)
    rows = []

    for summary_path in sorted(results_root.rglob("evaluation_summary.csv")):
        if any("backup" in part.lower() for part in summary_path.parts):
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        summary = summary.copy()
        summary["result_dir"] = str(summary_path.parent)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-run evaluation_summary.csv files into one table."
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("counterfactual_results"),
        help="Root directory containing counterfactual result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation_summaries.csv"),
        help="CSV path for the collected summary table.",
    )
    args = parser.parse_args()

    collected = collect_evaluation_summaries(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    collected.to_csv(args.output, index=False)
    print(f"Collected {len(collected)} evaluation summaries into {args.output}")


if __name__ == "__main__":
    main()
