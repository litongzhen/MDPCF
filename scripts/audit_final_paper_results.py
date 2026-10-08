"""Recompute delivered yield and paired ablation differences from frozen results."""

import csv
import json
from pathlib import Path
import statistics


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    repo = Path(__file__).resolve().parents[1]
    per = read_csv(repo / "results/final_main_ablation_3000/per_seed.csv")
    baselines = read_csv(repo / "results/final_baselines_multiseed_3000/all_methods.csv")
    assert len(per) == 45 and len(baselines) == 75
    groups = {}
    for row in per:
        assert row["black_box_seed"] == "42"
        groups.setdefault((row["dataset"], row["variant"]), []).append(int(row["seed"]))
    assert all(sorted(seeds) == [42, 43, 44] for seeds in groups.values())
    baseline_groups = {}
    for row in baselines:
        baseline_groups.setdefault((row["dataset"], row["method"]), []).append(row)
    assert len(baseline_groups) == 25
    assert all(len(rows) == 3 for rows in baseline_groups.values())
    yield_values = {}
    baseline_seeds = {}
    for row in per:
        summary = read_csv(repo / row["summary_path"])[0]
        assert int(float(summary["attempted_count"])) == 100
        assert int(float(summary["expected_cf_count"])) == 1000
        if row["variant"] == "csvae_transformer":
            value = float(summary["valid_cf_count"]) / float(summary["expected_cf_count"])
            yield_values.setdefault((row["dataset"], "MDPCE"), []).append(value)
    for row in baselines:
        assert int(float(row["attempted_count"])) == 100
        assert int(float(row["expected_cf_count"])) == 1000
        config_path = repo / row["result_dir"] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["black_box_seed"] == 42
        assert config["seed"] in (42, 43, 44)
        baseline_seeds.setdefault((row["dataset"], row["method"]), []).append(config["seed"])
        value = float(row["valid_cf_count"]) / float(row["expected_cf_count"])
        yield_values.setdefault((row["dataset"], row["method"]), []).append(value)
    assert all(sorted(seeds) == [42, 43, 44] for seeds in baseline_seeds.values())
    output = repo / "results/paper_revision_20261008"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "valid_cf_yield.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "n_seeds", "mean", "std"])
        writer.writeheader()
        for (dataset, method), values in sorted(yield_values.items()):
            writer.writerow(dict(dataset=dataset, method=method, n_seeds=len(values),
                                 mean=statistics.mean(values), std=statistics.stdev(values)))
    lookup = {(row["dataset"], row["variant"], row["seed"]): row for row in per}
    metrics = ["coverage", "mean_target_prob", "diversity", "proximity", "lof_log_scale"]
    with (output / "paired_ablation_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "comparator", "metric",
                                                    "mean_delta", "std_delta", "positive_seeds", "n_seeds"])
        writer.writeheader()
        for dataset in sorted({row["dataset"] for row in per}):
            for comparator in ("csvae_mlp", "standard_vae"):
                for metric in metrics:
                    deltas = [float(lookup[dataset, "csvae_transformer", seed][metric]) -
                              float(lookup[dataset, comparator, seed][metric]) for seed in ("42", "43", "44")]
                    writer.writerow(dict(dataset=dataset, comparator=comparator, metric=metric,
                                         mean_delta=statistics.mean(deltas), std_delta=statistics.stdev(deltas),
                                         positive_seeds=sum(d > 0 for d in deltas), n_seeds=3))
    report = {"main_runs": len(per), "baseline_runs": len(baselines), "black_box_seed": 42,
              "main_seed_groups": len(groups), "baseline_seed_groups": len(baseline_groups),
              "checks_passed": True}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
