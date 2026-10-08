import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_framework.evaluate import validate_constraints_for_evaluation


def infer_experiment_label(row):
    method = row["method"]
    result_dir = str(row.get("result_dir", ""))
    if method != "mdpcf":
        return method
    if "diversity_weight[0.0]" in result_dir:
        return "mdpcf_no_diversity"
    if "proximity_weight_latent" not in result_dir:
        return "mdpcf_no_latent_proximity"
    if "posthoc_sparsity[True]" not in result_dir:
        return "mdpcf_no_posthoc_sparsity"
    return "mdpcf_full"


def read_sampling_diagnostics(result_dir):
    diagnostics_path = Path(result_dir) / "sampling_diagnostics.csv"
    if not diagnostics_path.is_file():
        return {}
    diagnostics = pd.read_csv(diagnostics_path)
    if diagnostics.empty:
        return {}
    return diagnostics.iloc[0].to_dict()


def build_failure_analysis(summary_path):
    summaries = pd.read_csv(summary_path)
    rows = []

    for _, summary_row in summaries.iterrows():
        result_dir = Path(summary_row["result_dir"])
        cfs_path = result_dir / "cfs.csv"
        originals_path = result_dir / "original_test_samples.csv"
        info_path = Path("data") / summary_row["dataset"] / "info.json"
        if not cfs_path.is_file() or not originals_path.is_file() or not info_path.is_file():
            continue

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        cfs = pd.read_csv(cfs_path)
        originals = pd.read_csv(originals_path)
        originals_by_id = {row["id"]: row for _, row in originals.iterrows()}

        generated = len(cfs)
        expected = int(summary_row["expected_cf_count"])
        if generated:
            constraint_valid = validate_constraints_for_evaluation(
                cfs, originals_by_id, info
            )
            prob_valid = cfs["target_prob"].astype(float).to_numpy() > 0.5
            both_valid = prob_valid & constraint_valid
            failed_prob = int((~prob_valid).sum())
            failed_constraints = int((~constraint_valid).sum())
            failed_both = int(((~prob_valid) & (~constraint_valid)).sum())
            valid_count = int(both_valid.sum())
        else:
            failed_prob = 0
            failed_constraints = 0
            failed_both = 0
            valid_count = 0

        diagnostics = read_sampling_diagnostics(result_dir)
        row = {
            "dataset": summary_row["dataset"],
            "method": summary_row["method"],
            "experiment_label": infer_experiment_label(summary_row),
            "result_dir": str(result_dir),
            "attempted_count": int(summary_row["attempted_count"]),
            "expected_cf_count": expected,
            "generated_cf_count": generated,
            "valid_cf_count": valid_count,
            "missing_cf_count": max(expected - generated, 0),
            "failed_target_prob_count": failed_prob,
            "failed_constraints_count": failed_constraints,
            "failed_both_prob_and_constraints_count": failed_both,
            "failed_either_prob_or_constraints_count": generated - valid_count,
            "coverage": float(summary_row["coverage"]),
            "validity": float(summary_row["validity"]),
        }
        for key in [
            "optimized_valid_source_count",
            "optimized_valid_candidate_count",
            "post_projection_candidate_count",
            "post_projection_validity_count",
            "post_projection_constraints_count",
            "optimization_failed_count",
            "filtered_by_validity",
            "filtered_by_constraints",
        ]:
            row[f"mdpcf_{key}"] = diagnostics.get(key, "")
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build failure analysis from generated counterfactual results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/evaluation_summaries.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/failure_analysis_summary.csv"),
    )
    args = parser.parse_args()

    analysis = build_failure_analysis(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output, index=False)
    print(f"Wrote {len(analysis)} failure rows to {args.output}")


if __name__ == "__main__":
    main()
