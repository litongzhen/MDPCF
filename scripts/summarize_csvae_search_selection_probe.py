import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from scripts.summarize_csvae_search_parameter_probe import rank_configurations
except ModuleNotFoundError:
    from summarize_csvae_search_parameter_probe import rank_configurations


SELECTION_METRICS = ("dpp", "lof_dpp", "category_first")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def find_result(root, dataset, base_config_id, selection):
    result_root = Path(root) / dataset / base_config_id / selection
    matches = []
    for config_path in result_root.rglob("config.json"):
        evaluation_path = config_path.parent / "evaluation_summary.csv"
        manifest_path = config_path.parent / "source_manifest.csv"
        if evaluation_path.is_file() and manifest_path.is_file():
            matches.append((config_path, evaluation_path, manifest_path))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result for {dataset}/{base_config_id}/{selection}, "
            f"found {len(matches)}"
        )
    return matches[0]


def collect(root, selected_path):
    with Path(selected_path).open(encoding="utf-8") as handle:
        selected = json.load(handle)
    rows = []
    for dataset, base_config_ids in selected.items():
        for base_config_id in base_config_ids:
            for selection in SELECTION_METRICS:
                config_path, evaluation_path, manifest_path = find_result(
                    root, dataset, base_config_id, selection
                )
                with config_path.open(encoding="utf-8") as handle:
                    config = json.load(handle)
                with evaluation_path.open(newline="", encoding="utf-8") as handle:
                    evaluation = next(csv.DictReader(handle))
                rows.append(
                    {
                        "dataset": dataset,
                        "config_id": f"{base_config_id}_{selection}",
                        "base_config_id": base_config_id,
                        "selection_metric": selection,
                        "radius_scale": float(config["latent_init_radius_scale"]),
                        "diversity_weight": float(config["diversity_weight"]),
                        "coverage": float(evaluation["coverage"]),
                        "validity_expected": float(evaluation["validity_expected"]),
                        "generated_cf_count": int(evaluation["generated_cf_count"]),
                        "mean_target_prob": float(evaluation["mean_target_prob"]),
                        "diversity": float(evaluation["diversity"]),
                        "proximity": float(evaluation["proximity"]),
                        "lof_log_scale": float(evaluation["lof_log_scale"]),
                        "source_manifest_sha256": sha256(manifest_path),
                        "seed": int(config["seed"]),
                        "black_box_seed": int(config["black_box_seed"]),
                        "source_start": int(config["source_start"]),
                        "num_samples": int(config["num_samples"]),
                        "max_iter": int(config["max_iter"]),
                        "result_dir": str(config_path.parent.resolve()),
                    }
                )
    for dataset in selected:
        hashes = {
            row["source_manifest_sha256"]
            for row in rows
            if row["dataset"] == dataset
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Source manifest mismatch for stage B {dataset}")
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="results/csvae_search_parameter_probe/stage_b"
    )
    parser.add_argument(
        "--selected",
        default="results/csvae_search_parameter_probe/stage_b_selected.json",
    )
    parser.add_argument(
        "--output-root", default="results/csvae_search_parameter_probe"
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with Path(args.selected).open(encoding="utf-8") as handle:
        selected = json.load(handle)
    expected_rows = sum(len(config_ids) for config_ids in selected.values()) * len(
        SELECTION_METRICS
    )
    rows = collect(args.root, args.selected)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} stage-B rows, found {len(rows)}"
        )
    ranked = rank_configurations(rows)
    write_csv(output_root / "stage_b_summary.csv", rows)
    write_csv(output_root / "stage_b_ranking.csv", ranked)

    winners = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        candidates = [row for row in ranked if row["dataset"] == dataset]
        if not candidates:
            raise RuntimeError(f"No stage-B configuration passed gates for {dataset}")
        winner = candidates[0]
        winners[dataset] = {
            "base_config_id": winner["base_config_id"],
            "selection_metric": winner["selection_metric"],
            "radius_scale": winner["radius_scale"],
            "diversity_weight": winner["diversity_weight"],
        }
    with (output_root / "confirmation_selected.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(winners, handle, indent=2)
    print(json.dumps(winners, indent=2))


if __name__ == "__main__":
    main()
