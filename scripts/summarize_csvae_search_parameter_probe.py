import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("adult", "bank", "lending-club")
RADIUS_SCALES = (0.1, 0.2, 0.3)
DIVERSITY_WEIGHTS = (1.0, 2.0, 3.0)


def config_id(radius, diversity):
    return f"r{round(radius * 100):02d}_d{round(diversity * 10):02d}"


def _rank(values, reverse=False):
    ordered = sorted(set(values), reverse=reverse)
    return {value: index + 1 for index, value in enumerate(ordered)}


def rank_configurations(rows):
    eligible = [
        dict(row)
        for row in rows
        if float(row["coverage"]) >= 0.98
        and float(row["validity_expected"]) >= 0.90
    ]
    ranked = []
    for dataset in sorted({row["dataset"] for row in eligible}):
        group = [row for row in eligible if row["dataset"] == dataset]
        target_ranks = _rank(
            [float(row["mean_target_prob"]) for row in group], reverse=True
        )
        diversity_ranks = _rank(
            [float(row["diversity"]) for row in group], reverse=True
        )
        lof_ranks = _rank([float(row["lof_log_scale"]) for row in group])
        for row in group:
            row["target_rank"] = target_ranks[float(row["mean_target_prob"])]
            row["diversity_rank"] = diversity_ranks[float(row["diversity"])]
            row["lof_rank"] = lof_ranks[float(row["lof_log_scale"])]
            row["mean_rank"] = (
                row["target_rank"] + row["diversity_rank"] + row["lof_rank"]
            ) / 3.0
            ranked.append(row)
    return sorted(
        ranked,
        key=lambda row: (
            row["dataset"],
            row["mean_rank"],
            float(row["proximity"]),
            row["config_id"],
        ),
    )


def rank_fallback_configurations(rows):
    candidates = [dict(row) for row in rows if float(row["coverage"]) >= 0.90]
    ranked = []
    for dataset in sorted({row["dataset"] for row in candidates}):
        group = [row for row in candidates if row["dataset"] == dataset]
        yield_ranks = _rank(
            [float(row["validity_expected"]) for row in group], reverse=True
        )
        target_ranks = _rank(
            [float(row["mean_target_prob"]) for row in group], reverse=True
        )
        diversity_ranks = _rank(
            [float(row["diversity"]) for row in group], reverse=True
        )
        lof_ranks = _rank([float(row["lof_log_scale"]) for row in group])
        for row in group:
            row["yield_rank"] = yield_ranks[float(row["validity_expected"])]
            row["target_rank"] = target_ranks[float(row["mean_target_prob"])]
            row["diversity_rank"] = diversity_ranks[float(row["diversity"])]
            row["lof_rank"] = lof_ranks[float(row["lof_log_scale"])]
            row["fallback_mean_rank"] = (
                2 * row["yield_rank"]
                + row["target_rank"]
                + row["diversity_rank"]
                + row["lof_rank"]
            ) / 5.0
            ranked.append(row)
    return sorted(
        ranked,
        key=lambda row: (
            row["dataset"],
            row["fallback_mean_rank"],
            float(row["proximity"]),
            row["config_id"],
        ),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _find_evaluation(root, dataset, expected_id):
    config_root = Path(root) / dataset / expected_id
    matches = []
    for config_path in config_root.rglob("config.json"):
        evaluation_path = config_path.parent / "evaluation_summary.csv"
        manifest_path = config_path.parent / "source_manifest.csv"
        if evaluation_path.is_file() and manifest_path.is_file():
            matches.append((config_path, evaluation_path, manifest_path))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed evaluation for {dataset}/{expected_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def collect_stage_a(root, datasets=DATASETS):
    rows = []
    for dataset in datasets:
        for radius in RADIUS_SCALES:
            for diversity_weight in DIVERSITY_WEIGHTS:
                expected_id = config_id(radius, diversity_weight)
                config_path, evaluation_path, manifest_path = _find_evaluation(
                    root, dataset, expected_id
                )
                with config_path.open(encoding="utf-8") as handle:
                    config = json.load(handle)
                with evaluation_path.open(newline="", encoding="utf-8") as handle:
                    evaluation = next(csv.DictReader(handle))
                rows.append(
                    {
                        "dataset": dataset,
                        "config_id": expected_id,
                        "radius_scale": radius,
                        "diversity_weight": diversity_weight,
                        "coverage": float(evaluation["coverage"]),
                        "validity_expected": float(evaluation["validity_expected"]),
                        "generated_cf_count": int(evaluation["generated_cf_count"]),
                        "mean_target_prob": float(evaluation["mean_target_prob"]),
                        "diversity": float(evaluation["diversity"]),
                        "proximity": float(evaluation["proximity"]),
                        "lof_log_scale": float(evaluation["lof_log_scale"]),
                        "source_manifest_sha256": _sha256(manifest_path),
                        "seed": int(config["seed"]),
                        "black_box_seed": int(config["black_box_seed"]),
                        "source_start": int(config["source_start"]),
                        "num_samples": int(config["num_samples"]),
                        "max_iter": int(config["max_iter"]),
                        "result_dir": str(config_path.parent.resolve()),
                    }
                )
    for dataset in datasets:
        hashes = {
            row["source_manifest_sha256"]
            for row in rows
            if row["dataset"] == dataset
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Source manifest mismatch for {dataset}: {hashes}")
    return rows


def collect_named_configurations(root, dataset, config_ids):
    rows = []
    for expected_id in config_ids:
        config_path, evaluation_path, manifest_path = _find_evaluation(
            root, dataset, expected_id
        )
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        with evaluation_path.open(newline="", encoding="utf-8") as handle:
            evaluation = next(csv.DictReader(handle))
        rows.append(
            {
                "dataset": dataset,
                "config_id": expected_id,
                "radius_scale": float(config["latent_init_radius_scale"]),
                "diversity_weight": float(config["diversity_weight"]),
                "coverage": float(evaluation["coverage"]),
                "validity_expected": float(evaluation["validity_expected"]),
                "generated_cf_count": int(evaluation["generated_cf_count"]),
                "mean_target_prob": float(evaluation["mean_target_prob"]),
                "diversity": float(evaluation["diversity"]),
                "proximity": float(evaluation["proximity"]),
                "lof_log_scale": float(evaluation["lof_log_scale"]),
                "source_manifest_sha256": _sha256(manifest_path),
                "seed": int(config["seed"]),
                "black_box_seed": int(config["black_box_seed"]),
                "source_start": int(config["source_start"]),
                "num_samples": int(config["num_samples"]),
                "max_iter": int(config["max_iter"]),
                "result_dir": str(config_path.parent.resolve()),
            }
        )
    if len({row["source_manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError(f"Source manifest mismatch for {dataset} fallback")
    return rows


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-a-root",
        default="results/csvae_search_parameter_probe/stage_a",
    )
    parser.add_argument(
        "--output-root",
        default="results/csvae_search_parameter_probe",
    )
    parser.add_argument(
        "--bank-fallback-root",
        default="results/csvae_search_parameter_probe/stage_a_bank_mi3000",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = collect_stage_a(args.stage_a_root, args.datasets)
    ranked = rank_configurations(rows)
    _write_csv(output_root / "stage_a_summary.csv", rows)
    _write_csv(output_root / "stage_a_ranking.csv", ranked)

    fallback_path = output_root / "stage_a_fallback.json"
    fallback_rows = []
    if fallback_path.is_file() and Path(args.bank_fallback_root).is_dir():
        with fallback_path.open(encoding="utf-8") as handle:
            fallback_ids = json.load(handle).get("bank", [])
        if fallback_ids:
            fallback_rows = collect_named_configurations(
                args.bank_fallback_root, "bank", fallback_ids
            )
            fallback_ranked = rank_configurations(fallback_rows)
            _write_csv(output_root / "bank_mi3000_summary.csv", fallback_rows)
            _write_csv(output_root / "bank_mi3000_ranking.csv", fallback_ranked)

    selected = {}
    fallback = {}
    for dataset in args.datasets:
        source_ranking = (
            rank_configurations(fallback_rows)
            if dataset == "bank" and fallback_rows
            else ranked
        )
        selected[dataset] = [
            row["config_id"]
            for row in source_ranking
            if row["dataset"] == dataset
        ][:2]
        if len(selected[dataset]) != 2:
            fallback_ranked = rank_fallback_configurations(
                [row for row in rows if row["dataset"] == dataset]
            )
            fallback[dataset] = [row["config_id"] for row in fallback_ranked[:3]]
    if fallback:
        with (output_root / "stage_a_fallback.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(fallback, handle, indent=2)
        raise RuntimeError(
            "Hard gates left fewer than two configurations for: "
            + ", ".join(fallback)
            + ". Run stage_a_fallback.json at max_iter=3000."
        )
    with (output_root / "stage_b_selected.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
