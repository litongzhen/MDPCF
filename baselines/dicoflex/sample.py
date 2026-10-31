import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mdpcf.csvae.model import BBMLPCLF
from mdpcf.latent_utils import get_input_generate
from mdpcf.black_box_preprocessing import (
    BlackBoxPreprocessor, assert_encoded_input_alignment,
    guard_preprocessing_output, save_source_manifest,
)
from mdpcf.sample import (
    apply_integer_constraints,
    apply_numerical_domain_constraints,
    apply_raw_actionability_constraints,
    recover_data_from_encoded,
    select_source_indices,
)
from utils_train import get_black_box_model_path, preprocess, save_run_config


def load_official_dicoflex():
    repo_root = Path(__file__).resolve().parents[2]
    dicoflex_root = repo_root / "baselines" / "DiCoFlex-main"
    if not dicoflex_root.exists():
        raise FileNotFoundError(f"Official DiCoFlex source tree not found: {dicoflex_root}")

    dicoflex_root_str = str(dicoflex_root)
    if dicoflex_root_str in sys.path:
        sys.path.remove(dicoflex_root_str)
    sys.path.insert(0, dicoflex_root_str)
    return dicoflex_root


load_official_dicoflex()

from counterfactuals.datasets.generic_counterfactual import (  # noqa: E402
    MulticlassCounterfactualWrapper,
    generate_multiclass_counterfactuals,
    train_multiclass_counterfactual_flow_model,
)
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow  # noqa: E402


class MDPCFBlackBoxAdapter:
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device

    def predict_proba(self, x):
        if torch.is_tensor(x):
            tensor = x.float().to(self.device)
            return_tensor = True
        else:
            tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            return_tensor = False

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)

        if return_tensor:
            return probs
        return probs.detach().cpu().numpy()

    def predict(self, x):
        probs = self.predict_proba(x)
        if torch.is_tensor(probs):
            return torch.argmax(probs, dim=1)
        return np.argmax(probs, axis=1)


def project_encoded_features(encoded_data, info):
    encoded_data = np.asarray(encoded_data, dtype=np.float32).copy()
    d_numerical = len(info["num_col_idx"])

    if d_numerical:
        encoded_data[:, :d_numerical] = np.clip(encoded_data[:, :d_numerical], 0.0, 1.0)

    start = d_numerical
    for cat_col_idx in info["cat_col_idx"]:
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        width = len(info["column_domain"][col_name])
        block = encoded_data[:, start : start + width]
        hard = np.zeros_like(block)
        valid_rows = np.isfinite(block).all(axis=1)
        if valid_rows.any():
            hard[np.arange(len(block))[valid_rows], block[valid_rows].argmax(axis=1)] = 1.0
        encoded_data[:, start : start + width] = hard
        start += width

    return encoded_data


def build_output_dir(args):
    output_folder_name = (
        f"{args.dataname}/DiCoFlex/"
        f"epochs[{args.dicoflex_epochs}]_p[{args.dicoflex_p_value}]_"
        f"temp[{args.dicoflex_temperature}]_nn[{args.dicoflex_n_nearest}]_"
        f"train[{args.dicoflex_train_samples}]"
    )
    if args.num_samples > 0:
        output_folder_name = f"{output_folder_name}/{args.num_samples}_samples"
    else:
        output_folder_name = f"{output_folder_name}/all_samples"
    if args.total_CFs > 1:
        output_folder_name = f"{output_folder_name}_CFs_{args.total_CFs}"
    return os.path.join(args.save_path, output_folder_name)


def load_black_box(args, data_dir, input_shape):
    model_filename = get_black_box_model_path(args, data_dir, cat_encoding="one-hot")
    model = BBMLPCLF(input_shape).to(args.device)
    model.load_state_dict(torch.load(model_filename, map_location=args.device))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, model_filename


def subsample_training_rows(X_train, y_train, max_rows, seed=42):
    if max_rows <= 0 or len(X_train) <= max_rows:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    selected = []
    classes = np.unique(y_train)
    per_class = max(1, max_rows // len(classes))

    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        take = min(per_class, len(cls_indices))
        selected.extend(rng.choice(cls_indices, size=take, replace=False).tolist())

    remaining = max_rows - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        pool = np.array([idx for idx in range(len(y_train)) if idx not in selected_set])
        if len(pool) > 0:
            selected.extend(
                rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist()
            )

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)
    return X_train[selected], y_train[selected]


def main(args):
    dataname = args.dataname
    data_dir = f"data/{dataname}"
    save_path = build_output_dir(args)
    os.makedirs(save_path, exist_ok=True)
    guard_preprocessing_output(args, save_path)
    save_run_config(args, save_path)

    generation_inputs = get_input_generate(args)
    info = generation_inputs["info"]
    num_inverse = generation_inputs["num_inverse"]
    cat_inverse = generation_inputs["cat_inverse"]

    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    column_names.remove(target_col)

    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )
    X_train_encoded, X_test_encoded = [
        np.asarray(x, dtype=np.float32) for x in X_num_cat_ohe
    ]
    raw_preprocessor = BlackBoxPreprocessor(data_dir)
    test_df = pd.read_csv(f"data/{dataname}/test.csv")
    assert_encoded_input_alignment(raw_preprocessor.transform(test_df), X_test_encoded)

    black_box, model_filename = load_black_box(args, data_dir, X_train_encoded.shape[1])
    classifier = MDPCFBlackBoxAdapter(black_box, args.device)
    print("Loaded black box model from", model_filename)

    train_probs = classifier.predict_proba(X_train_encoded)
    test_probs = classifier.predict_proba(X_test_encoded)
    y_train_pred = np.argmax(train_probs, axis=1).astype(np.int64)
    test_preds = np.argmax(test_probs, axis=1).astype(np.int64)

    X_train_flow, y_train_flow = subsample_training_rows(
        X_train_encoded,
        y_train_pred,
        max_rows=args.dicoflex_train_samples,
        seed=42,
    )
    print(f"DiCoFlex flow training rows: {len(X_train_flow)} / {len(X_train_encoded)}")

    target_class_index = int(args.target_class)
    indices_negative_class = select_source_indices(
        test_preds, target_class_index, args.num_samples, getattr(args, "source_start", 0)
    )
    save_source_manifest(save_path, indices_negative_class, test_probs, target_class_index)

    if len(np.unique(y_train_flow)) < 2:
        raise ValueError("DiCoFlex needs at least two predicted classes in the training set.")

    features = X_train_encoded.shape[1]
    masks = np.ones((1, features), dtype=np.float32) * 1e-3
    p_values = [float(args.dicoflex_p_value)]

    dataset_cf = MulticlassCounterfactualWrapper(
        X=X_train_flow,
        y=y_train_flow,
        factual_classes=np.unique(y_train_flow),
        p_values=p_values,
        masks=masks,
        n_nearest=args.dicoflex_n_nearest,
        noise_level=0.01,
        classifier=classifier,
        prob_threshold=args.dicoflex_prob_threshold,
        log_level="INFO",
        numerical_pos=len(info["num_col_idx"]),
    )

    model_dir = os.path.join(save_path, "dicoflex_model")
    load_existing = os.path.exists(os.path.join(model_dir, "flow_model.pth"))
    start_time = time.time()
    flow_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset_cf,
        flow_model_class=MaskedAutoregressiveFlow,
        mask_features=len(masks),
        hidden_features=64,
        num_layers=5,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=256,
        num_epochs=args.dicoflex_epochs,
        patience=50,
        noise_level=0.01,
        device=args.device,
        save_dir=model_dir,
        log_interval=10,
        balanced=True,
        load_from_save_dir=load_existing,
    )

    train_time = time.time() - start_time
    print(f"DiCoFlex flow ready in {train_time:.2f}s")

    test_df = pd.read_csv(f"data/{dataname}/test.csv")
    negative_test_df = test_df.drop(target_col, axis=1).iloc[indices_negative_class].reset_index(drop=True)
    negative_test_df["id"] = list(range(len(negative_test_df)))
    negative_test_df = negative_test_df[["id"] + column_names]

    all_cfs = []
    factual_points = X_test_encoded[indices_negative_class]

    for source_id, encoded_source in tqdm(
        enumerate(factual_points),
        total=len(factual_points),
        desc="DiCoFlex CFs for test set",
    ):
        generated_cfs, _ = generate_multiclass_counterfactuals(
            model=flow_model,
            factual_points=encoded_source.reshape(1, -1),
            target_class=target_class_index,
            p_value=float(args.dicoflex_p_value),
            mask=np.array([1.0], dtype=np.float32),
            n_samples=args.total_CFs,
            temperature=args.dicoflex_temperature,
            device=args.device,
            num_classes=len(dataset_cf.classes),
        )

        encoded_cfs = generated_cfs.reshape(args.total_CFs, -1)
        encoded_cfs = encoded_cfs[np.isfinite(encoded_cfs).all(axis=1)]
        if len(encoded_cfs) == 0:
            continue

        encoded_cfs = project_encoded_features(encoded_cfs, info)
        cfs = recover_data_from_encoded(encoded_cfs, info, num_inverse, cat_inverse)
        cfs.columns = column_names
        original_row = negative_test_df.drop(columns=["id"]).iloc[source_id]
        cfs = apply_raw_actionability_constraints(cfs, original_row, info)
        cfs = apply_numerical_domain_constraints(cfs, info)
        cfs = apply_integer_constraints(cfs, info)

        final_encoded = raw_preprocessor.transform(cfs)
        target_probs = classifier.predict_proba(final_encoded)[:, target_class_index]
        cfs["id"] = source_id
        cfs["target_prob"] = target_probs
        all_cfs.append(cfs[["id"] + column_names + ["target_prob"]])

    cols = ["id"] + column_names
    if len(all_cfs) == 0:
        print("No DiCoFlex counterfactuals generated for current settings.")
        syn_df = pd.DataFrame(columns=cols + ["target_prob"])
    else:
        syn_df = pd.concat(all_cfs, ignore_index=True)
        syn_df = syn_df[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index=False)
    negative_test_df.to_csv(
        os.path.join(save_path, "original_test_samples.csv"), index=False
    )
    print("Saving sampled data to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation DiCoFlex")
    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
