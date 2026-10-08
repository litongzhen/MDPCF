"""Test deterministic constrained swaps on saved pools, without retraining."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def select_pareto_subset(distances, costs, probabilities, reference):
    distances = np.asarray(distances, dtype=float)
    costs = np.asarray(costs, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    selected = list(reference)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Reference must be a nonempty unique subset")
    if not (np.isfinite(distances).all() and np.isfinite(costs).all()
            and np.isfinite(probabilities).all()):
        raise ValueError("Metrics must be finite")
    cost_limits = costs[selected].sum(axis=0)
    probability_floor = probabilities[selected].sum()
    swaps = 0
    while True:
        current = float(distances[np.ix_(selected, selected)].sum() / 2)
        best = None
        for position in range(len(selected)):
            kept = selected[:position] + selected[position + 1:]
            for candidate in sorted(set(range(len(costs))) - set(selected)):
                trial = kept + [candidate]
                if (costs[trial].sum(axis=0) > cost_limits + 1e-10).any():
                    continue
                if probabilities[trial].sum() < probability_floor - 1e-10:
                    continue
                score = float(distances[np.ix_(trial, trial)].sum() / 2)
                if score > current + 1e-10 and (best is None or score > best[0]):
                    best = score, trial
        if best is None:
            return selected, swaps
        selected = best[1]
        swaps += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from evaluation_framework.evaluate import _mixed_distance, build_evaluation_summary

    args.output.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    for complete in sorted(args.root.glob("*/complete.json")):
        folder = complete.parent
        output = args.output / folder.name
        output.mkdir(exist_ok=True)
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
            raise ValueError("Candidate/source alignment mismatch")
        mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
        num = {mapping[i] for i in info.get("num_col_idx", [])}
        cat = {mapping[i] for i in info.get("cat_col_idx", [])}
        frames = []
        for pool_path, (source_id, group) in zip(pools, groups):
            pool = pd.read_csv(pool_path)
            features = [c for c in group.columns if c not in {"id", "target_prob"}]
            old = np.flatnonzero(pool.reference_selected.to_numpy() == 1).tolist()
            pd.testing.assert_frame_equal(
                pool.iloc[old][features].sort_values(features).reset_index(drop=True),
                group[features].sort_values(features).reset_index(drop=True),
                check_dtype=False, atol=1e-6, rtol=1e-6,
            )
            source = originals.loc[originals.id == source_id].iloc[0]
            distances = np.zeros((len(pool), len(pool)))
            proximity = []
            for i in range(len(pool)):
                proximity.append(_mixed_distance(pool.iloc[i], source, features, num, cat, info)["diversity"])
                for j in range(i):
                    distances[i, j] = distances[j, i] = _mixed_distance(
                        pool.iloc[i], pool.iloc[j], features, num, cat, info
                    )["diversity"]
            costs = np.column_stack([pool.selector_lof, proximity])
            indices, swaps = select_pareto_subset(distances, costs, pool.target_probability, old)
            selected = pool.iloc[indices][features].copy()
            selected["id"] = source_id
            selected["target_prob"] = pool.iloc[indices].target_probability.to_numpy()
            frames.append(selected)
            diagnostics.append({"dataset": dataset, "variant": config["latent_model"],
                                "source_id": source_id, "swaps": swaps,
                                "candidate_count": len(pool)})
        alternative = pd.concat(frames, ignore_index=True)
        alternative.to_csv(output / "pareto_cfs.csv", index=False)
        for label, cfs in (("reference", reference), ("pareto", alternative)):
            summary = build_evaluation_summary(cfs, originals, info,
                total_CFs=config["total_CFs"], train_df=train, enforce_constraints=False)
            summary.to_csv(output / f"{label}_evaluation_summary.csv", index=False)
            metric = summary.iloc[0]
            rows.append({"dataset": dataset, "variant": config["latent_model"],
                "selector": label, "seed": config["seed"], "source_start": config["source_start"],
                **{key: metric[key] for key in ("coverage", "validity_generated", "validity_expected",
                    "mean_target_prob", "diversity", "proximity", "lof_log_scale")}})
    pd.DataFrame(rows).to_csv(args.output / "shared_evaluation.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "per_source_swaps.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "input": str(args.root), "rule": "strict_reference_budget_single_swap",
        "costs": ["selector_lof_proxy", "mixed_proximity"],
        "probability": "at_least_reference_mean", "objective": "mixed_diversity",
        "formal_results_changed": False, "uses_saved_valid_candidate_pools": True,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
