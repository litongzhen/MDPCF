"""Re-evaluate saved paired selections with the shared final-paper metrics."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from evaluation_framework.evaluate import build_evaluation_summary

    rows = []
    for complete in sorted(args.root.glob("*/complete.json")):
        folder = complete.parent
        config = json.loads((folder / "probe_config.json").read_text(encoding="utf-8"))
        dataset = config["dataname"]
        info = json.loads((repo / "data" / dataset / "info.json").read_text(encoding="utf-8"))
        train = pd.read_csv(repo / "data" / dataset / "train.csv")
        run = next((folder / "reference_run").rglob("original_test_samples.csv")).parent
        originals = pd.read_csv(run / "original_test_samples.csv")
        reference = pd.read_csv(run / "cfs.csv")
        groups = list(reference.groupby("id", sort=False))
        pools = sorted(folder.glob("pool_*.csv"))
        if len(groups) != len(pools):
            raise ValueError("Candidate-pool/source alignment mismatch")
        budget_frames = []
        for pool_path, (source_id, group) in zip(pools, groups):
            pool = pd.read_csv(pool_path)
            feature_names = [c for c in group.columns if c not in {"id", "target_prob"}]
            old = pool.loc[pool.reference_selected == 1, feature_names].reset_index(drop=True)
            # Reference selector order can differ, so compare rows after sorting all features.
            pd.testing.assert_frame_equal(
                old.sort_values(feature_names).reset_index(drop=True),
                group[feature_names].sort_values(feature_names).reset_index(drop=True),
                check_dtype=False, atol=1e-6, rtol=1e-6,
            )
            selected = pool.loc[pool.budget_selected == 1, feature_names].copy()
            selected["id"] = source_id
            selected["target_prob"] = pool.loc[pool.budget_selected == 1, "target_probability"].values
            budget_frames.append(selected)
        budget = pd.concat(budget_frames, ignore_index=True)
        budget.to_csv(folder / "budget_cfs.csv", index=False)
        for selector, cfs in (("reference", reference), ("budget", budget)):
            summary = build_evaluation_summary(
                cfs, originals, info, total_CFs=config["total_CFs"], train_df=train,
                enforce_constraints=False,
            )
            summary.to_csv(folder / f"{selector}_evaluation_summary.csv", index=False)
            row = summary.iloc[0]
            rows.append({"dataset": dataset, "variant": config["latent_model"],
                         "selector": selector, "seed": config["seed"],
                         "source_start": config["source_start"],
                         **{key: row[key] for key in (
                             "attempted_count", "valid_cf_count", "coverage", "validity_generated",
                             "validity_expected", "mean_target_prob", "diversity", "proximity",
                             "lof_log_scale",
                         )}})
    pd.DataFrame(rows).to_csv(args.root / "shared_evaluation.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
