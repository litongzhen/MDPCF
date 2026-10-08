import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdpcf.csvae.model import BBMLPCLF
from utils_train import get_black_box_model_path, preprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a fixed counterfactual experiment protocol manifest."
    )
    parser.add_argument("--dataname", default="adult")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--hidden_dims", type=int, default=16)
    parser.add_argument("--num_encoding", default="minmax")
    parser.add_argument("--output_dir", default="results/protocol")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path("data") / args.dataname
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    X_num_cat_ohe, _, _, _, _ = preprocess(
        str(data_dir),
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding=args.num_encoding,
    )

    model_path = get_black_box_model_path(args, str(data_dir))
    model = BBMLPCLF(X_num_cat_ohe[0].shape[-1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_num_cat_ohe[1], dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)

    negative_indices = np.where(preds != args.target_class)[0]
    if args.num_samples > 0:
        negative_indices = negative_indices[: args.num_samples]

    manifest = pd.DataFrame(
        {
            "selected_order": np.arange(len(negative_indices), dtype=int),
            "test_index": negative_indices.astype(int),
            "black_box_pred": preds[negative_indices].astype(int),
            "target_prob": probs[negative_indices, args.target_class],
        }
    )
    manifest_path = (
        output_dir / f"{args.dataname}_black_box_negative_first_{len(manifest)}.csv"
    )
    manifest.to_csv(manifest_path, index=False)

    config = {
        "dataname": args.dataname,
        "num_samples_requested": args.num_samples,
        "num_samples_selected": int(len(manifest)),
        "target_class": args.target_class,
        "hidden_dims": args.hidden_dims,
        "num_encoding": args.num_encoding,
        "selection_rule": "first N test instances whose black-box prediction is not target_class",
        "black_box_model_path": model_path,
        "manifest_path": str(manifest_path),
    }
    config_path = output_dir / f"{args.dataname}_protocol.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Protocol manifest: {manifest_path}")
    print(f"Protocol config: {config_path}")


if __name__ == "__main__":
    main()
