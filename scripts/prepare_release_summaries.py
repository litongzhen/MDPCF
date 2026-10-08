"""Export curated formal evidence without raw samples or machine-local paths."""

import argparse
import json
from pathlib import Path
import re

import pandas as pd


def sanitize_config(config, cli_keys):
    return {key: value for key, value in config.items()
            if key in cli_keys and key != "save_path" and value is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("release"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    summaries = output / "summaries"
    configs = output / "configs"
    summaries.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    cli_keys = set(re.findall(r'"--([a-zA-Z0-9_]+)"', (repo / "utils.py").read_text(encoding="utf-8")))
    main = pd.read_csv(repo / "results/final_main_ablation_3000/per_seed.csv")
    baseline = pd.read_csv(repo / "results/final_baselines_multiseed_3000/all_methods.csv")
    if len(main) != 45 or len(baseline) != 75:
        raise ValueError("Expected complete 45/75 formal runs")
    for _, row in main.iterrows():
        source = repo / Path(row.summary_path).parent / "config.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        name = f"{row.dataset}-{row.variant}-seed{int(row.seed)}.json"
        (configs / name).write_text(json.dumps(sanitize_config(config, cli_keys), indent=2), encoding="utf-8")
    seeds = []
    for _, row in baseline.iterrows():
        config = json.loads((repo / Path(row.result_dir) / "config.json").read_text(encoding="utf-8"))
        seeds.append(config["seed"])
        name = f"{row.dataset}-{row.method}-seed{config['seed']}.json"
        (configs / name).write_text(json.dumps(sanitize_config(config, cli_keys), indent=2), encoding="utf-8")
    baseline["seed"] = seeds
    main.drop(columns=[c for c in main.columns if c.endswith("_path") or c.endswith("_dir")]).to_csv(
        summaries / "main_per_seed.csv", index=False)
    baseline.drop(columns=[c for c in baseline.columns if c.endswith("_path") or c.endswith("_dir")]).to_csv(
        summaries / "baselines_per_seed.csv", index=False)
    pd.read_csv(repo / "results/final_main_ablation_3000/mean_std.csv").to_csv(
        summaries / "main_mean_std.csv", index=False)
    (output / "evidence_manifest.json").write_text(json.dumps({
        "main_runs": len(main), "baseline_runs": len(baseline), "seeds": [42, 43, 44],
        "black_box_seed": 42, "includes_raw_counterfactuals": False,
        "includes_checkpoints": False, "includes_exploratory_probes": False,
        "note": "Saved configuration files omit save_path; use a new output directory.",
    }, indent=2), encoding="utf-8")
    print(f"Exported {len(main)} main and {len(baseline)} baseline runs to {output}")


if __name__ == "__main__":
    main()
