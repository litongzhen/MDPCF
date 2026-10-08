import sys
from pathlib import Path


def load_official_dice():
    repo_root = Path(__file__).resolve().parents[2]
    dice_root = repo_root / "baselines" / "dice" / "DiCE-official"
    if not dice_root.exists():
        raise FileNotFoundError(f"Official DiCE source tree not found: {dice_root}")

    for module_name in list(sys.modules):
        if module_name == "dice_ml" or module_name.startswith("dice_ml."):
            sys.modules.pop(module_name)

    dice_root_str = str(dice_root)
    if dice_root_str in sys.path:
        sys.path.remove(dice_root_str)
    sys.path.insert(0, str(dice_root))
    return dice_root


load_official_dice()

import dice_ml
import torch
import argparse
import warnings
import time
import joblib
import torch.nn.functional as F

from tqdm import tqdm
from contextlib import redirect_stdout
import io

from utils_train import (
    preprocess,
    TabularDataset,
    get_name_form_args,
    get_optim_params_form_args,
    get_black_box_model_path,
    save_run_config,
)

from mdpcf.csvae.model import BBMLPCLF
from mdpcf.latent_utils import get_input_generate, recover_data, split_num_cat_target
from mdpcf.sample import select_source_indices
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from dice_ml.utils import helpers

warnings.filterwarnings("ignore")
from raiutils.exceptions import UserConfigValidationException


def build_dice_actionability_args(original_instance, info, feature_names):
    immutable_features = set(info.get("immutable", []))
    features_to_vary = [
        feature for feature in feature_names if feature not in immutable_features
    ]
    permitted_range = {}
    row = original_instance.iloc[0]
    column_domain = info.get("column_domain", {})

    for feature in info.get("monotone_increase", []):
        if feature in feature_names:
            upper = column_domain.get(feature, [None, row[feature]])[1]
            permitted_range[feature] = [row[feature], upper]

    for feature in info.get("monotone_decrease", []):
        if feature in feature_names:
            lower = column_domain.get(feature, [row[feature], None])[0]
            permitted_range[feature] = [lower, row[feature]]

    ordinal = info.get("ordinal", {})
    for feature in info.get("ordinal_monotone_increase", []):
        if feature in feature_names and feature in ordinal:
            ordered_values = ordinal[feature]
            current_value = row[feature]
            if current_value in ordered_values:
                current_rank = ordered_values.index(current_value)
                permitted_range[feature] = ordered_values[current_rank:]

    for feature in info.get("ordinal_monotone_decrease", []):
        if feature in feature_names and feature in ordinal:
            ordered_values = ordinal[feature]
            current_value = row[feature]
            if current_value in ordered_values:
                current_rank = ordered_values.index(current_value)
                permitted_range[feature] = ordered_values[: current_rank + 1]

    args = {"features_to_vary": features_to_vary}
    if permitted_range:
        args["permitted_range"] = permitted_range
    return args


def project_integer_features(cfs, info):
    cfs = cfs.copy()
    column_domain = info.get("column_domain", {})

    for feature in info.get("integer", []):
        if feature not in cfs.columns:
            continue

        values = pd.to_numeric(cfs[feature], errors="coerce").round()
        domain = column_domain.get(feature)
        if isinstance(domain, list) and len(domain) == 2:
            values = values.clip(lower=float(domain[0]), upper=float(domain[1]))
        cfs[feature] = values

    return cfs


def get_target_probabilities(dice_model, cfs, target_class_index):
    outputs = np.asarray(dice_model.get_output(cfs))

    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)

    if outputs.shape[1] == 1:
        probs_1 = torch.sigmoid(
            torch.tensor(outputs[:, 0], dtype=torch.float32)
        ).numpy()
        probs = np.stack([1.0 - probs_1, probs_1], axis=1)
    elif (
        np.any(outputs < 0)
        or np.any(outputs > 1)
        or not np.allclose(outputs.sum(axis=1), 1.0, atol=1e-4)
    ):
        probs = torch.softmax(torch.tensor(outputs, dtype=torch.float32), dim=1).numpy()
    else:
        probs = outputs

    return probs[:, target_class_index]


def main(args):
    # ML_modelpath = helpers.get_adult_income_modelpath(backend="PYT")
    # # Step 2: dice_ml.Model
    # # m = dice_ml.Model(model_path=ML_modelpath, backend="PYT", func="ohe-min-max")
    # model = torch.load(ML_modelpath)

    # print(model)
    # exit(1)

    dataname = args.dataname
    data_dir = f"data/{dataname}"

    device = args.device
    num_samples = args.num_samples
    verbose = args.verbose

    save_path = args.save_path
    dice_method = args.dice_method
    dice_post_hoc_sparsity = args.dice_post_hoc_sparsity

    proximity_weight_input = args.proximity_weight_input
    diversity_weight = args.diversity_weight

    total_CFs = args.total_CFs
    min_iter = args.min_iter
    max_iter = args.max_iter
    hidden_dims = args.hidden_dims

    plot_gradients = args.plot_gradients

    optimization_parameters = get_optim_params_form_args(args)

    output_folder_name = f"{dataname}/DiCE/{dice_method}/{optimization_parameters}"

    if num_samples > 0:
        output_folder_name = f"{output_folder_name}/{num_samples}_samples"
    else:
        output_folder_name = f"{output_folder_name}/all_samples"

    if total_CFs > 1:
        output_folder_name = f"{output_folder_name}_CFs_{total_CFs}"

    # if validity:
    #     output_folder_name += "_validity"
    # if proximity:
    #     output_folder_name += "_proximity"
    # if sparsity:
    #     output_folder_name += "_sparsity"

    save_path = os.path.join(save_path, output_folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_run_config(args, save_path)

    generation_inputs = get_input_generate(args)
    info = generation_inputs["info"]
    num_inverse = generation_inputs["num_inverse"]
    cat_inverse = generation_inputs["cat_inverse"]
    y = generation_inputs["y"]

    target_class = info["target_class"]
    negative_class = info["negative_class"]

    model_filename = get_black_box_model_path(
        args, data_dir, cat_encoding="one-hot"
    )

    # input_shape = len(info["num_col_idx"]) + len(info["cat_col_idx"])

    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )

    input_shape = X_num_cat_ohe[0].shape[-1]

    dice_device = "cpu"
    loaded_black_box_clf = BBMLPCLF(input_shape).to(dice_device)

    loaded_black_box_clf.load_state_dict(torch.load(model_filename, map_location="cpu"))
    # Ensure black_box_clf parameters are not updated
    for param in loaded_black_box_clf.parameters():
        param.requires_grad = False

    loaded_black_box_clf.eval()

    target_class_index = int(args.target_class)

    train_set_path = f"data/{dataname}/train.csv"
    test_set_path = f"data/{dataname}/test.csv"

    train_df = pd.read_csv(train_set_path)
    test_df = pd.read_csv(test_set_path)

    print("Loaded black box model from", model_filename)

    # input_col_order = info["input_col_order"]
    target_column = info["target_col"]
    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    print(target_col)
    print(column_names)
    column_names.remove(target_col)

    num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
    cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

    # -------------------------------- DiCE stuff -------------------------------------------

    dice_args = {
        "desired_class": target_class_index,
        "verbose": False,
        "total_CFs": total_CFs,
    }
    if dice_method == "gradient":
        dice_args.update(
            {
                "proximity_weight": proximity_weight_input,
                "diversity_weight": diversity_weight,
                "min_iter": min_iter,
                "max_iter": max_iter,
            }
        )
    elif dice_method == "genetic":
        dice_args["proximity_weight"] = proximity_weight_input
        dice_args["diversity_weight"] = diversity_weight
        dice_args["maxiterations"] = max_iter

    if not dice_post_hoc_sparsity:
        dice_args["posthoc_sparsity_param"] = None

    print("DiCE Args", dice_args)

    train_df[target_column] = train_df[target_column].map(
        {target_class: 1, negative_class: 0}
    )

    data_dice = dice_ml.data.Data(
        dataframe=train_df, outcome_name=target_column, continuous_features=num_cols
    )
    dice_model = dice_ml.Model(
        loaded_black_box_clf,
        backend="PYT",
        model_type="classifier",
        func="ohe-min-max",
    )
    exp = dice_ml.Dice(data_dice, dice_model, method=dice_method)

    # -----------------------------------------------------------------------------------------
    test_df = test_df.drop(target_column, axis=1)

    test_preds = np.argmax(dice_model.get_output(test_df), axis=1)

    source_start = int(getattr(args, "source_start", 0))
    indices_negative_class = select_source_indices(
        test_preds, target_class_index, num_samples, source_start
    )

    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)

    """
        Generating samples    
    """
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()

    start_time = time.time()

    all_cfs = []

    for indx, row in tqdm(
        negative_test_df.iterrows(),
        total=len(negative_test_df),
        desc="CFs for test set",
    ):

        original_instance = negative_test_df.iloc[indx : indx + 1]

        dice_call_args = dice_args.copy()
        actionability_args = build_dice_actionability_args(original_instance, info, column_names)
        if "permitted_range" in actionability_args:
            permitted_range = data_dice.permitted_range.copy()
            permitted_range.update(actionability_args["permitted_range"])
            actionability_args["permitted_range"] = permitted_range
        dice_call_args.update(actionability_args)

        try:
            dice_exp = exp.generate_counterfactuals(original_instance, **dice_call_args)
        except (
            UserConfigValidationException
        ) as e:  # cf not found, just add the original sample, and set target_prob to 0.0
            # original_instance["id"] = indx
            # original_instance["target_prob"] = 0.0
            # all_cfs.append(original_instance)
            continue

        cfs = dice_exp.cf_examples_list[0].final_cfs_df[column_names]
        cfs = project_integer_features(cfs, info)

        # cf = cf[input_col_order]
        target_probs_final = get_target_probabilities(
            dice_model, cfs, target_class_index
        )
        cfs["id"] = indx

        cfs["target_prob"] = target_probs_final

        all_cfs.append(cfs)

    # profiler.stop()

    # profiler.print()

    cols = ["id"] + column_names

    if len(all_cfs) == 0:
        print("No valid DiCE counterfactuals generated for current settings.")
        syn_df = pd.DataFrame(columns=cols + ["target_prob"])
    else:
        syn_df = pd.concat(all_cfs, ignore_index=True)
        syn_df = syn_df[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index=False)

    end_time = time.time()
    print("Time:", end_time - start_time)

    negative_test_df["id"] = list(range(len(negative_test_df)))

    negative_test_df = negative_test_df[cols]

    negative_test_df.to_csv(
        os.path.join(save_path, "original_test_samples.csv"), index=False
    )

    # print('Saving sampled data to {}'.format(save_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generation DiCE")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
