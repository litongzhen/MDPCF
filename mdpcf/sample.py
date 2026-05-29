import torch
import argparse
import warnings
import time

from tqdm import tqdm

from utils_train import (
    preprocess,
    get_name_form_args,
    get_optim_params_form_args,
)

from mdpcf.vae.model import BBMLPCLF
from mdpcf.latent_utils import (
    get_input_generate,
    recover_data,
    split_num_cat_target,
    generate_equiangular_unit_vectors,
)
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")


class Decoder_Black_Box(torch.nn.Module):
    def __init__(
        self,
        black_box_clf,
        pre_decoder,
        input_encoded_continuous_feature_indexes,
        input_encoded_categorical_feature_indexes,
    ):
        super(Decoder_Black_Box, self).__init__()
        self.black_box_clf = black_box_clf
        self.pre_decoder = pre_decoder
        self.input_encoded_continuous_feature_indexes = (
            input_encoded_continuous_feature_indexes
        )
        self.num_numerical = len(input_encoded_continuous_feature_indexes)
        self.input_encoded_categorical_feature_indexes = (
            input_encoded_categorical_feature_indexes
        )
        self.fixed_w = None
        self.immutable_mask = None
        self.immutable_values = None

    def set_fixed_w(self, w):
        self.fixed_w = w.detach()

    def set_immutable_reference(self, immutable_mask, immutable_values):
        if not torch.is_tensor(immutable_mask):
            immutable_mask = torch.tensor(immutable_mask, dtype=torch.bool)
        if not torch.is_tensor(immutable_values):
            immutable_values = torch.tensor(immutable_values, dtype=torch.float32)
        self.immutable_mask = immutable_mask.detach()
        self.immutable_values = immutable_values.detach()

    def _apply_immutable_constraints(self, decoded):
        if self.immutable_mask is None or self.immutable_values is None:
            return decoded
        immutable_mask = self.immutable_mask.to(decoded.device)
        immutable_values = self.immutable_values.to(decoded.device)
        if immutable_mask.dim() == 1:
            immutable_mask = immutable_mask.unsqueeze(0)
        if immutable_values.dim() == 1:
            immutable_values = immutable_values.unsqueeze(0)
        if immutable_mask.shape[0] != decoded.shape[0]:
            immutable_mask = immutable_mask.repeat(decoded.shape[0], 1)
        if immutable_values.shape[0] != decoded.shape[0]:
            immutable_values = immutable_values.repeat(decoded.shape[0], 1)
        return torch.where(immutable_mask, immutable_values, decoded)

    def bb_only(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor
        # Pass concatenated output through pre_decoder
        output = self.black_box_clf(input_data)

        if original_input_dim == 1:
            return output[0]

        return output

    def decode_z(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        if self.fixed_w is None:
            raise ValueError("fixed_w must be set before decoding z")
        fixed_w = self.fixed_w.detach()
        if fixed_w.dim() == 1:
            fixed_w = fixed_w.unsqueeze(0)
        if fixed_w.shape[0] != input_data.shape[0]:
            fixed_w = fixed_w.repeat(input_data.shape[0], 1)
        x_hat_num, x_hat_cat = self.pre_decoder(input_data, fixed_w)

        # print(x_hat_num)
        # for xx in x_hat_cat:
        #     print(xx)

        # exit(1)
        processed_cats = []
        for cat_pred in x_hat_cat:
            if cat_pred.dim() == 2:
                processed_cats.append(torch.softmax(cat_pred, dim=-1))
            else:
                processed_cats.append(cat_pred)
        concatenated = torch.cat((x_hat_num, *processed_cats), dim=-1)
        return self._apply_immutable_constraints(concatenated)

    def forward(self, input_data):

        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        if self.fixed_w is None:
            raise ValueError("fixed_w must be set before forward")
        fixed_w = self.fixed_w.detach()
        if fixed_w.dim() == 1:
            fixed_w = fixed_w.unsqueeze(0)
        if fixed_w.shape[0] != input_data.shape[0]:
            fixed_w = fixed_w.repeat(input_data.shape[0], 1)
        x_hat_num, x_hat_cat = self.pre_decoder(input_data, fixed_w)

        # print(x_hat_num)
        # for xx in x_hat_cat:
        #     print(xx)

        # exit(1)
        processed_cats = []
        for cat_pred in x_hat_cat:
            if cat_pred.dim() == 2:
                processed_cats.append(torch.softmax(cat_pred, dim=-1))
            else:
                processed_cats.append(cat_pred)
        concatenated = torch.cat((x_hat_num, *processed_cats), dim=-1)
        concatenated = self._apply_immutable_constraints(concatenated)

        output = self.black_box_clf(concatenated)

        if original_input_dim == 1:
            return output[0]

        return output


def optimize_z_counterfactuals(
    z0,
    w0,
    decoder_black_box_clf,
    total_CFs,
    min_iter,
    max_iter,
    proximity_weight_latent,
    proximity_weight_input,
    diversity_weight,
    latent_init_radius,
    original_input_instance,
    immutable_mask,
    device,
):
    latent_dim = z0.shape[-1]
    directions = generate_equiangular_unit_vectors(total_CFs, d=latent_dim)
    directions = torch.tensor(directions, dtype=torch.float32, device=device)
    z_candidates = (
        z0.unsqueeze(0) + latent_init_radius * directions
    ).clone().detach()
    z_candidates.requires_grad_(True)
    for param in decoder_black_box_clf.parameters():
        param.requires_grad_(False)
    decoder_black_box_clf.set_fixed_w(w0)
    decoder_black_box_clf.set_immutable_reference(immutable_mask, original_input_instance)
    optimizer = torch.optim.Adam([z_candidates], lr=0.05)
    target = torch.ones(total_CFs, dtype=torch.long, device=device)
    original_input_instance = torch.tensor(
        original_input_instance, dtype=torch.float32, device=device
    ).repeat(total_CFs, 1)

    best_valid = None
    best_score = None
    best_decoded = None

    for step in range(max_iter):
        optimizer.zero_grad()
        decoded = decoder_black_box_clf.decode_z(z_candidates)
        logits = decoder_black_box_clf.bb_only(decoded)
        validity_loss = torch.nn.functional.cross_entropy(logits, target)
        latent_prox = ((z_candidates - z0.unsqueeze(0)) ** 2).mean()
        input_prox = ((decoded - original_input_instance) ** 2).mean()
        if total_CFs > 1:
            diversity_loss = -torch.pdist(z_candidates, p=2).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=device)
        loss = (
            validity_loss
            + proximity_weight_latent * latent_prox
            + proximity_weight_input * input_prox
            + diversity_weight * diversity_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)[:, 1]
            mask = probs > 0.5
            if mask.any():
                score = probs[mask].mean().item()
                if best_score is None or score > best_score:
                    best_score = score
                    best_valid = z_candidates[mask].detach().cpu()
                    best_decoded = decoded[mask].detach().cpu()
        if step >= min_iter and best_valid is not None:
            break

    if best_valid is None:
        return None, None
    return best_valid, best_decoded


def main(args):

    dataname = args.dataname
    data_dir = f"data/{dataname}"

    device = args.device
    num_samples = args.num_samples
    save_path = args.save_path
    verbose = args.verbose

    # steps = args.steps
    # validity = args.validity
    # proximity = args.proximity
    # sparsity = args.sparsity
    # latent_clf = args.latent_clf
    # diffusion = args.diffusion
    proximity_weight_input = args.proximity_weight_input
    proximity_weight_latent = args.proximity_weight_latent

    hidden_dims = args.hidden_dims

    optimization_parameters = get_optim_params_form_args(args)

    dice_method = args.dice_method

    num_encoding = args.num_encoding

    total_CFs = args.total_CFs
    min_iter = args.min_iter
    max_iter = args.max_iter

    output_file_name_vae = get_name_form_args(args)

    output_folder_parts = [
        dataname,
        "MDPCF",
        output_file_name_vae,
        dice_method,
        optimization_parameters,
    ]
    if num_samples > 0:
        output_folder_parts.append(f"{num_samples}_samples")
    else:
        output_folder_parts.append("all_samples")
    if total_CFs > 1:
        output_folder_parts[-1] = f"{output_folder_parts[-1]}_CFs_{total_CFs}"

    save_path = os.path.join(save_path, *output_folder_parts)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if os.listdir(save_path):
            print("CFs for setting", output_folder_name, "already computed")
            exit(0)

    (
        train_z,
        test_z,
        train_w,
        test_w,
        curr_dir,
        dataset_dir,
        ckpt_path,
        vae_ckpt_dir,
        info,
        num_inverse,
        cat_inverse,
        y,
    ) = get_input_generate(args)

    test_set_path = f"data/{dataname}/test.csv"

    test_df = pd.read_csv(test_set_path)

    column_names = info["column_names"]
    target_col = info["target_col"]
    column_names.remove(target_col)

    test_z_df = pd.DataFrame(test_z)
    test_w_df = pd.DataFrame(test_w, columns=[f"w_{i}" for i in range(test_w.shape[1])])

    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )
    X_num_raw, X_cat_raw, categories_raw, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        num_encoding="minmax",
    )

    """
        Generating samples    
    """
    start_time = time.time()

    model_filename = f"{data_dir}/black_box_mlp.pkl"

    if hidden_dims is not None:
        model_filename = f"{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl"

    input_shape = X_num_cat_ohe[0].shape[-1]

    loaded_black_box_clf = BBMLPCLF(input_shape).to(device)

    loaded_black_box_clf.load_state_dict(torch.load(model_filename, map_location=device))
    # Ensure black_box_clf parameters are not updated
    for param in loaded_black_box_clf.parameters():
        param.requires_grad = False

    loaded_black_box_clf.eval()

    pre_decoder = info["pre_decoder"].to(device)
    pre_decoder.eval()

    test_tensor_input = torch.tensor(
        X_num_cat_ohe[1], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        test_preds = torch.argmax(loaded_black_box_clf(test_tensor_input), dim=1).cpu().numpy()

    input_encoded_continuous_feature_indexes = list(range(X_num_cat_ohe[0].shape[1]))
    input_encoded_categorical_feature_indexes = []

    immutable_cols = set(info.get("immutable", []))
    immutable_mask_np = np.zeros(X_num_cat_ohe[0].shape[1], dtype=bool)
    immutable_mask_np[0] = "age" in immutable_cols
    start = len(info["num_col_idx"])
    for local_idx, cat_col_idx in enumerate(info["cat_col_idx"]):
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        cat_info = categories_raw[local_idx]
        cat_width = int(cat_info) if isinstance(cat_info, (int, np.integer)) else len(cat_info)
        if col_name in immutable_cols:
            immutable_mask_np[start : start + cat_width] = True
        start += cat_width

    decoder_black_box_clf = Decoder_Black_Box(
        loaded_black_box_clf,
        pre_decoder,
        input_encoded_continuous_feature_indexes,
        input_encoded_categorical_feature_indexes,
    ).to(device)

    decoder_black_box_clf.eval()

    target_class_index = 1

    print("Loaded black box model from", model_filename)

    indices_negative_class = np.where(test_preds != target_class_index)[0]
    if num_samples > 0:
        indices_negative_class = indices_negative_class[:num_samples]

    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)

    negative_test_z_df = test_z_df.iloc[indices_negative_class].reset_index(drop=True)
    negative_test_w_df = test_w_df.iloc[indices_negative_class].reset_index(drop=True)

    negative_test_encoded = X_num_cat_ohe[1][indices_negative_class]
    immutable_ref_np = negative_test_encoded.copy()

    print(
        {
            "total_CFs": total_CFs,
            "min_iter": min_iter,
            "max_iter": max_iter,
            "proximity_weight_input": proximity_weight_input,
            "proximity_weight_latent": proximity_weight_latent,
            "diversity_weight": getattr(args, "diversity_weight", 1.0),
            "latent_init_radius": getattr(args, "latent_init_radius", 0.5),
        }
    )

    all_cfs = []

    diversity_weight = getattr(args, "diversity_weight", 1.0)
    latent_init_radius = getattr(args, "latent_init_radius", 0.5)

    for indx, row in tqdm(
        negative_test_z_df.iterrows(),
        total=len(negative_test_z_df),
        desc="CFs for test set",
    ):
        z0 = torch.tensor(row.to_numpy(), dtype=torch.float32, device=device)
        w0 = torch.tensor(
            negative_test_w_df.iloc[indx].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        z_valid, decoded_valid = optimize_z_counterfactuals(
            z0,
            w0,
            decoder_black_box_clf,
            total_CFs,
            min_iter,
            max_iter,
            max(proximity_weight_latent, 0.0),
            max(proximity_weight_input, 0.0),
            diversity_weight,
            latent_init_radius,
            immutable_ref_np[indx : indx + 1],
            torch.tensor(immutable_mask_np, dtype=torch.bool, device=device),
            device,
        )
        if z_valid is None:
            continue
        w_valid = w0.unsqueeze(0).repeat(z_valid.shape[0], 1).detach().cpu().numpy()
        latent_pairs = np.concatenate([z_valid.cpu().numpy(), w_valid], axis=1)
        syn_num, syn_cat = split_num_cat_target(
            latent_pairs, info, num_inverse, cat_inverse, args.device
        )
        cfs = recover_data(syn_num, syn_cat, info)
        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        cfs.rename(columns=idx_name_mapping, inplace=True)
        for immutable_col in info.get("immutable", []):
            if immutable_col in cfs.columns:
                cfs[immutable_col] = negative_test_df.loc[indx, immutable_col]
        with torch.no_grad():
            probs = torch.softmax(
                decoder_black_box_clf.bb_only(decoded_valid.to(device)),
                dim=-1,
            )[:, target_class_index]
        target_probs_final = probs.detach().cpu().numpy()
        cfs["id"] = indx
        cfs["target_prob"] = target_probs_final
        all_cfs.append(cfs)

    if len(all_cfs) == 0:
        print("No valid counterfactuals generated for current settings.")
        syn_df = pd.DataFrame(columns=["id"] + info["column_names"] + ["target_prob"])
    else:
        syn_df = pd.concat(all_cfs, ignore_index=True)

    syn_df.reset_index(drop=True, inplace=True)

    cols = ["id"] + info["column_names"]

    if len(all_cfs) > 0:
        syn_df = syn_df[cols + ["target_prob"]]

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index=False)

    end_time = time.time()
    print("Time:", end_time - start_time)

    negative_test_df["id"] = list(range(len(negative_test_df)))
    negative_test_df = negative_test_df[cols]

    negative_test_df.to_csv(
        os.path.join(save_path, "original_test_samples.csv"), index=False
    )

    print("Saving sampled data to {}".format(save_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generation")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
