import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_DEFAULTS = {
    "adult": (" >50K", " <=50K"),
    "bank": ("yes", "no"),
    "lending-club": ("Fully Paid", "Charged Off"),
    "default": (1, 0),
    "gmc": (1, 0),
}


def infer_feature_types(info, columns):
    target_col = info["target_col"]
    if info.get("num_col_idx") is not None and info.get("cat_col_idx") is not None:
        num_col_idx = list(info["num_col_idx"])
        cat_col_idx = list(info["cat_col_idx"])
        target_col_idx = list(info.get("target_col_idx", [columns.index(target_col)]))
    else:
        num_col_idx = [
            idx
            for idx, col in enumerate(columns)
            if info["column_info"][col] == "float"
        ]
        cat_col_idx = [
            idx
            for idx, col in enumerate(columns)
            if info["column_info"][col] == "str" and col != target_col
        ]
        target_col_idx = [idx for idx, col in enumerate(columns) if col == target_col]
    return num_col_idx, cat_col_idx, target_col_idx


def build_mappings(columns, num_col_idx, cat_col_idx):
    idx_mapping = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)
    for idx in range(len(columns)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {int(v): int(k) for k, v in idx_mapping.items()}
    idx_name_mapping = {int(idx): col for idx, col in enumerate(columns)}
    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def prepare_dataset(name, source_root, target_root):
    source_dir = source_root / name
    target_dir = target_root / name
    info_path = target_root / "Info" / f"{name}.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    train_df = pd.read_csv(source_dir / "train.csv")
    test_df = pd.read_csv(source_dir / "test.csv")
    ignore_columns = list(info.get("ignore_columns", []))
    if ignore_columns:
        train_df = train_df.drop(columns=[col for col in ignore_columns if col in train_df.columns])
        test_df = test_df.drop(columns=[col for col in ignore_columns if col in test_df.columns])

    target_col = info["target_col"]
    train_df = train_df[[col for col in train_df.columns if col != target_col] + [target_col]]
    test_df = test_df[[col for col in test_df.columns if col != target_col] + [target_col]]
    columns = train_df.columns.tolist()

    num_col_idx, cat_col_idx, target_col_idx = infer_feature_types(info, columns)
    num_columns = [columns[idx] for idx in num_col_idx]
    cat_columns = [columns[idx] for idx in cat_col_idx]
    target_columns = [columns[idx] for idx in target_col_idx]

    for df in (train_df, test_df):
        if num_columns:
            df[num_columns] = df[num_columns].replace("?", np.nan).astype(np.float32)
        if cat_columns:
            df[cat_columns] = df[cat_columns].replace("?", "nan").astype(str)

    target_dir.mkdir(parents=True, exist_ok=True)
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    X_cat_test = test_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    np.save(target_dir / "X_num_train.npy", X_num_train)
    np.save(target_dir / "X_num_test.npy", X_num_test)
    np.save(target_dir / "X_cat_train.npy", X_cat_train)
    np.save(target_dir / "X_cat_test.npy", X_cat_test)
    np.save(target_dir / "y_train.npy", y_train)
    np.save(target_dir / "y_test.npy", y_test)

    train_df.to_csv(target_dir / "train.csv", index=False)
    test_df.to_csv(target_dir / "test.csv", index=False)

    idx_mapping, inverse_idx_mapping, idx_name_mapping = build_mappings(
        columns, num_col_idx, cat_col_idx
    )
    column_domain = {}
    for col in num_columns:
        column_domain[col] = [float(train_df[col].min()), float(train_df[col].max())]
    for col in cat_columns:
        column_domain[col] = train_df[col].dropna().unique().tolist()

    target_class, negative_class = TARGET_DEFAULTS.get(name, (1, 0))
    info.update(
        {
            "column_names": columns,
            "data_path": str(target_dir / f"{name}.csv"),
            "test_path": "",
            "target_class": target_class,
            "negative_class": negative_class,
            "column_domain": column_domain,
            "num_max": dict(zip(num_columns, X_num_train.max(axis=0).astype(str))),
            "num_min": dict(zip(num_columns, X_num_train.min(axis=0).astype(str))),
            "train_num": int(len(train_df)),
            "test_num": int(len(test_df)),
            "input_col_order": num_columns + cat_columns,
            "idx_mapping": idx_mapping,
            "inverse_idx_mapping": inverse_idx_mapping,
            "idx_name_mapping": idx_name_mapping,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "metadata": {
                "columns": {
                    **{
                        int(idx): {
                            "sdtype": "numerical",
                            "computer_representation": "Float",
                        }
                        for idx in num_col_idx
                    },
                    **{int(idx): {"sdtype": "categorical"} for idx in cat_col_idx},
                    **{int(idx): {"sdtype": "categorical"} for idx in target_col_idx},
                }
            },
        }
    )

    with (target_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(
        f"{name}: train={len(train_df)} test={len(test_df)} "
        f"num={len(num_columns)} cat={len(cat_columns)} target={target_columns}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="DiCoFlex-main/data")
    parser.add_argument("--target-root", default="data")
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    for dataset in args.datasets:
        prepare_dataset(dataset, source_root, target_root)


if __name__ == "__main__":
    main()
