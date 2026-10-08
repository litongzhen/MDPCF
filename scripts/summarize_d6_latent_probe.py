import argparse
import csv
from pathlib import Path


DATASETS = ("adult", "default", "gmc")
VARIANTS = ("csvae", "standard_vae")


def find_evaluation(root, dataset, variant):
    matches = list((Path(root) / dataset / variant).rglob("evaluation_summary.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one D6 evaluation for {dataset}/{variant}, found {len(matches)}"
        )
    return matches[0]


def collect(root):
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            evaluation_path = find_evaluation(root, dataset, variant)
            config_path = evaluation_path.parent / "config.json"
            with evaluation_path.open(newline="", encoding="utf-8") as handle:
                evaluation = next(csv.DictReader(handle))
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "coverage": float(evaluation["coverage"]),
                    "validity_expected": float(evaluation["validity_expected"]),
                    "mean_target_prob": float(evaluation["mean_target_prob"]),
                    "diversity": float(evaluation["diversity"]),
                    "proximity": float(evaluation["proximity"]),
                    "lof_log_scale": float(evaluation["lof_log_scale"]),
                    "config_path": str(config_path.resolve()),
                    "result_dir": str(evaluation_path.parent.resolve()),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/d6_latent_probe/runs")
    parser.add_argument("--output", default="results/d6_latent_probe/summary.csv")
    args = parser.parse_args()
    rows = collect(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
