"""Compare selectors on identical saved candidate pools without changing generation."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np


def select_budget_subset(distances, lof_scores, probabilities, size, tolerance=0.1):
    distances = np.asarray(distances, dtype=float)
    scores = np.asarray(lof_scores, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if size < 1 or size > len(scores) or tolerance < 0:
        raise ValueError("Invalid subset size or LOF tolerance")
    if not np.isfinite(scores).all() or (scores <= 0).any():
        raise ValueError("LOF scores must be finite and positive")
    budget = float(np.sort(scores)[:size].sum() * (1 + tolerance))
    selected = [min(range(len(scores)), key=lambda i: (scores[i], -probabilities[i], i))]
    remaining = set(range(len(scores))) - set(selected)
    while len(selected) < size:
        # Reserve the cheapest possible completion so every choice remains feasible.
        feasible = []
        for i in sorted(remaining):
            slots = size - len(selected) - 1
            completion = sorted(scores[j] for j in remaining if j != i)[:slots]
            if scores[selected].sum() + scores[i] + sum(completion) <= budget + 1e-10:
                feasible.append(i)
        if not feasible:
            raise RuntimeError("No feasible completion within LOF budget")
        chosen = max(feasible, key=lambda i: (
            float(distances[i, selected].sum()),
            float(distances[i, selected].min()),
            probabilities[i], -scores[i], -i,
        ))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.1)
    options = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import mdpcf.sample as sampling
    from utils import get_args, prepare_runtime_args

    config = json.loads(options.config.read_text(encoding="utf-8"))
    options.output.mkdir(parents=True, exist_ok=True)
    cli = []
    known_options = set(re.findall(r'"(--[a-zA-Z0-9_]+)"', (repo / "utils.py").read_text(encoding="utf-8")))
    for key, value in config.items():
        if value is not None and f"--{key}" in known_options:
            cli.extend([f"--{key}", str(value)])
    cli.extend([
        "--mode", "sample", "--save_path", str(options.output / "reference_run"),
        "--source_start", str(options.source_start), "--num_samples", str(options.num_samples),
    ])
    sys.argv = ["main.py"] + cli
    args = prepare_runtime_args(get_args())
    manifest = dict(vars(args))
    manifest.update(probe_tolerance=options.tolerance, reference_config=str(options.config),
                    selector_probe="same_pool_lof_budget_distance", changes_generation=False)
    (options.output / "probe_config.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    original_selector = sampling.select_diverse_counterfactual_subset
    rows = []

    def compare(cfs, latents, probs, info, total_CFs, **kwargs):
        result = original_selector(cfs, latents, probs, info, total_CFs, **kwargs)
        scores = np.asarray(kwargs.get("candidate_plausibility_scores"), dtype=float)
        if len(scores.shape) != 1 or len(scores) != len(cfs):
            raise ValueError("Probe requires LOF-scored candidate pools")
        features = sampling._selection_feature_names(info, cfs)
        mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
        num = {mapping[i] for i in info.get("num_col_idx", [])}
        cat = {mapping[i] for i in info.get("cat_col_idx", [])}
        distances = np.zeros((len(cfs), len(cfs)))
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                distances[i, j] = distances[j, i] = sampling._selection_mixed_distance(
                    cfs.iloc[i], cfs.iloc[j], features, num, cat, info
                )
        old_indices = [cfs.index.get_loc(i) for i in result[0].index]
        new_indices = select_budget_subset(
            distances, scores, probs, min(total_CFs, len(cfs)), options.tolerance
        )
        call = len(rows) + 1
        pool = cfs.copy()
        pool["target_probability"] = probs
        pool["selector_lof"] = scores
        pool["reference_selected"] = [int(i in old_indices) for i in range(len(cfs))]
        pool["budget_selected"] = [int(i in new_indices) for i in range(len(cfs))]
        pool.to_csv(options.output / f"pool_{call:03d}.csv", index=False)
        row = {"pool": call, "candidate_count": len(cfs), "selected_count": len(new_indices)}
        for label, indices in (("reference", old_indices), ("budget", new_indices)):
            sub = distances[np.ix_(indices, indices)]
            pairs = sub[np.triu_indices(len(indices), 1)]
            row[f"{label}_diversity"] = float(pairs.mean()) if len(pairs) else float("nan")
            row[f"{label}_lof"] = float(scores[indices].mean())
            row[f"{label}_target"] = float(np.asarray(probs)[indices].mean())
        rows.append(row)
        with (options.output / "paired_selection.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"SELECTION_PROBE pool={call} " + json.dumps(row), flush=True)
        return result

    sampling.select_diverse_counterfactual_subset = compare
    try:
        sampling.main(args)
    finally:
        sampling.select_diverse_counterfactual_subset = original_selector
    (options.output / "complete.json").write_text(
        json.dumps({"completed": True, "paired_pools": len(rows),
                    "attempted_sources": options.num_samples}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
