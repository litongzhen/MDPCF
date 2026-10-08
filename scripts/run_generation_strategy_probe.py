"""Run a bounded, resumable equal-budget generation strategy exploration."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd


def strategy_overrides(config):
    return {
        "reference": {},
        "radius_half": {"latent_init_radius_scale": config["latent_init_radius_scale"] / 2},
        "proximity_double": {"proximity_weight_input": config["proximity_weight_input"] * 2},
        "trajectory": {"trajectory_quality_selection": True},
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/generation_strategy_probe_20261008"))
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = options.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(repo / "results/final_main_ablation_3000/per_seed.csv")
    known = set(re.findall(r'"(--[a-zA-Z0-9_]+)"', (repo / "utils.py").read_text(encoding="utf-8")))
    jobs = []
    for dataset in ("gmc", "lending-club"):
        for variant in ("csvae_transformer", "standard_vae"):
            match = summary[(summary.dataset == dataset) & (summary.variant == variant) & (summary.seed == 42)]
            if len(match) != 1:
                raise ValueError(f"Expected one reference run: {dataset}/{variant}")
            config_path = repo / Path(match.iloc[0].summary_path).parent / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for strategy, overrides in strategy_overrides(config).items():
                folder = root / f"{dataset}-{variant}-{strategy}"
                actual = {k: v for k, v in config.items() if f"--{k}" in known and v is not None}
                actual.update(overrides)
                actual.update(mode="sample", method="mdpcf", source_start=100, num_samples=3,
                              save_path=str(folder / "run"), candidate_CFs=50, total_CFs=10,
                              min_iter=500, max_iter=3000)
                jobs.append((folder, dataset, variant, strategy, actual))
    protocol = {"seed": 42, "black_box_seed": 42, "source_start": 100,
                "num_samples": 3, "total_jobs": len(jobs), "max_iter": 3000,
                "candidate_CFs": 50, "retrain": False,
                "scope": "tuning sources only; fresh-source confirmation required",
                "strategies": list(strategy_overrides(config))}
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing protocol differs")
    write_json(protocol_path, protocol)
    rows = []
    for index, (folder, dataset, variant, strategy, actual) in enumerate(jobs, 1):
        folder.mkdir(exist_ok=True)
        manifest = folder / "probe_config.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != actual:
            raise ValueError(f"Existing configuration differs: {folder}")
        write_json(manifest, actual)
        complete = folder / "complete.json"
        if complete.exists() and not options.resume:
            raise ValueError("Completed results exist; use --resume")
        if not complete.exists():
            for stage in ("sample", "evaluate"):
                write_json(root / "status.json", {"stage": stage, "job": index,
                    "total_jobs": len(jobs), "completed_jobs": len(rows),
                    "dataset": dataset, "variant": variant, "strategy": strategy})
                cli = [sys.executable, "-u", "main.py"]
                for key, value in {**actual, "mode": stage}.items():
                    cli.extend([f"--{key}", str(value)])
                with (folder / f"{stage}.log").open("a", encoding="utf-8") as log:
                    process = subprocess.run(cli, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    write_json(root / "status.json", {"stage": "failed", "job": index,
                        "completed_jobs": len(rows), "dataset": dataset, "variant": variant,
                        "strategy": strategy, "failed_stage": stage, "exit_code": process.returncode})
                    raise RuntimeError(f"{folder.name}/{stage} failed; see log")
            artifacts = list((folder / "run").rglob("evaluation_summary.csv"))
            if len(artifacts) != 1:
                raise ValueError(f"Expected one evaluation artifact in {folder}")
            write_json(complete, {"summary_path": str(artifacts[0])})
        artifact = json.loads(complete.read_text(encoding="utf-8"))["summary_path"]
        metric = pd.read_csv(artifact).iloc[0]
        if int(metric.attempted_count) != 3 or int(metric.expected_cf_count) != 30:
            raise ValueError("Evaluation source or output budget mismatch")
        rows.append({"dataset": dataset, "variant": variant, "strategy": strategy,
            "summary_path": artifact, **{k: metric[k] for k in (
                "coverage", "validity_generated", "validity_expected", "mean_target_prob",
                "diversity", "proximity", "lof_log_scale")}})
        pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
        print(f"COMPLETE {index}/{len(jobs)} {folder.name}", flush=True)
    write_json(root / "status.json", {"stage": "complete", "completed_jobs": len(rows),
                                     "total_jobs": len(jobs)})


if __name__ == "__main__":
    main()
