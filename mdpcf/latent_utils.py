import json
import os
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess, get_name_form_args
from mdpcf.csvae.model import (
    ConditionalVAEDecoderModel,
    Decoder_model,
    StandardVAEDecoderModel,
)

LATENT_ARTIFACT_PROTOCOL = "cruds_aligned_v3"
STANDARD_VAE_ARTIFACT_PROTOCOL = "standard_vae_v1"
CONDITIONAL_VAE_ARTIFACT_PROTOCOL = "conditional_vae_v1"
LATENT_ARTIFACT_METADATA = "latent_artifact_protocol.json"


def validate_latent_artifact_protocol(
    csvae_ckpt_dir,
    latent_model="csvae",
    w_structure=None,
    w_dim=None,
    token_dim=None,
    latent_dim=None,
):
    metadata_path = os.path.join(csvae_ckpt_dir, LATENT_ARTIFACT_METADATA)
    if not os.path.isfile(metadata_path):
        raise RuntimeError(
            f"Latent arrays lack the required {LATENT_ARTIFACT_PROTOCOL} protocol marker."
        )
    with open(metadata_path, "r") as handle:
        metadata = json.load(handle)
    expected_protocols = {
        "csvae": LATENT_ARTIFACT_PROTOCOL,
        "standard_vae": STANDARD_VAE_ARTIFACT_PROTOCOL,
        "cvae": CONDITIONAL_VAE_ARTIFACT_PROTOCOL,
    }
    if latent_model not in expected_protocols:
        raise ValueError(f"Unknown latent_model: {latent_model}")
    expected = expected_protocols[latent_model]
    if metadata.get("latent_artifact_protocol") != expected:
        raise RuntimeError(
            f"Latent arrays must use {expected}; found "
            f"{metadata.get('latent_artifact_protocol', 'missing')}."
        )
    if latent_model == "csvae" and w_structure is not None:
        if metadata.get("w_structure") != w_structure:
            raise RuntimeError(
                f"Latent arrays use w_structure={metadata.get('w_structure')}; "
                f"expected {w_structure}."
            )
    if latent_model == "csvae" and w_dim is not None:
        if int(metadata.get("w_dim", -1)) != int(w_dim):
            raise RuntimeError(
                f"Latent arrays use w_dim={metadata.get('w_dim')}; expected {w_dim}."
            )
    expected_token_dim = 4 if token_dim is None else int(token_dim)
    expected_latent_dim = 4 if latent_dim is None else int(latent_dim)
    actual_token_dim = int(metadata.get("token_dim", 4))
    actual_latent_dim = int(metadata.get("latent_dim", 4))
    if (actual_token_dim, actual_latent_dim) != (
        expected_token_dim,
        expected_latent_dim,
    ):
        raise RuntimeError(
            "Latent arrays use dimensions "
            f"token={actual_token_dim}, z={actual_latent_dim}; expected "
            f"token={expected_token_dim}, z={expected_latent_dim}."
        )


def load_test_latent_arrays(
    csvae_ckpt_dir, latent_model="csvae", w_structure=None, w_dim=None,
    token_dim=None, latent_dim=None,
):
    validate_latent_artifact_protocol(
        csvae_ckpt_dir, latent_model, w_structure=w_structure, w_dim=w_dim,
        token_dim=token_dim, latent_dim=latent_dim,
    )
    test_z = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "test_z.npy"))).float()
    if latent_model in {"standard_vae", "cvae"}:
        test_w = torch.zeros_like(test_z)
    else:
        test_w = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "test_w.npy"))).float()
    return test_z, test_w


def load_train_latent_arrays(
    csvae_ckpt_dir, latent_model="csvae", w_structure=None, w_dim=None,
    token_dim=None, latent_dim=None,
):
    validate_latent_artifact_protocol(
        csvae_ckpt_dir, latent_model, w_structure=w_structure, w_dim=w_dim,
        token_dim=token_dim, latent_dim=latent_dim,
    )
    train_z = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "train_z.npy"))).float()
    if latent_model in {"standard_vae", "cvae"}:
        train_w = torch.zeros_like(train_z)
    else:
        train_w = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "train_w.npy"))).float()
    return train_z, train_w


def get_input_generate(args):
    dataname = args.dataname
    num_encoding = args.num_encoding

    # activations
    sigmoid = args.sigmoid
    gumbel_softmax = args.gumbel_softmax
    tau = args.tau
    csvae_backbone = getattr(args, "csvae_backbone", "transformer")
    mlp_parameter_match = getattr(args, "mlp_parameter_match", False)
    latent_model = getattr(args, "latent_model", "csvae")
    w_structure = getattr(args, "w_structure", "global")
    w_dim = getattr(args, "w_dim", 2)
    token_dim = int(getattr(args, "token_dim", 4))
    latent_dim = int(getattr(args, "latent_dim", 4))

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    output_file_name_csvae = get_name_form_args(args)

    dataset_dir = f"data/{dataname}"

    decoder_save_path = (
        f"{curr_dir}/csvae/ckpt/{dataname}/{output_file_name_csvae}/decoder.pt"
    )

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    task_type = info["task_type"]

    csvae_ckpt_dir = f"{curr_dir}/csvae/ckpt/{dataname}/{output_file_name_csvae}"

    _, _, categories, d_numerical, num_inverse, cat_inverse, y = preprocess(
        dataset_dir, task_type=task_type, inverse=True, num_encoding=num_encoding
    )

    if args.method == "mdpcf":
        train_z, train_w = load_train_latent_arrays(
            csvae_ckpt_dir,
            latent_model,
            w_structure=w_structure,
            w_dim=w_dim,
            token_dim=token_dim,
            latent_dim=latent_dim,
        )
        test_z, test_w = load_test_latent_arrays(
            csvae_ckpt_dir,
            latent_model,
            w_structure=w_structure,
            w_dim=w_dim,
            token_dim=token_dim,
            latent_dim=latent_dim,
        )

        decoder_classes = {
            "csvae": Decoder_model,
            "standard_vae": StandardVAEDecoderModel,
            "cvae": ConditionalVAEDecoderModel,
        }
        decoder_class = decoder_classes[latent_model]
        pre_decoder = decoder_class(
            2, d_numerical, categories, token_dim, n_head=1, factor=32,
            gumbel_softmax=gumbel_softmax, sigmoid=sigmoid, tau=tau,
            backbone=csvae_backbone,
            **(
                {
                    "dim_w": int(test_w.shape[-1]),
                    "mlp_parameter_match": mlp_parameter_match,
                }
                if latent_model == "csvae"
                else {}
            ),
        )

        pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location="cpu"))

        info["pre_decoder"] = pre_decoder
        info["latent_shape"] = list(test_z.shape[1:])
        info["w_shape"] = list(test_w.shape[1:])
        info["latent_dim"] = int(np.prod(test_z.shape[1:]))
        info["w_dim"] = int(np.prod(test_w.shape[1:]))

        return {
            "train_z": train_z,
            "train_w": train_w,
            "test_z": test_z,
            "test_w": test_w,
            "w_prior_mu": (
                torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "w_prior_mu.npy"))).float()
                if latent_model == "csvae"
                and os.path.isfile(os.path.join(csvae_ckpt_dir, "w_prior_mu.npy"))
                else None
            ),
            "info": info,
            "num_inverse": num_inverse,
            "cat_inverse": cat_inverse,
            "y": y,
        }

    return {
        "info": info,
        "num_inverse": num_inverse,
        "cat_inverse": cat_inverse,
        "y": y,
    }


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    pre_decoder = info["pre_decoder"].to(device)
    z_dim = info["latent_dim"]
    w_dim = info["w_dim"]
    syn_tensor = torch.tensor(syn_data, dtype=torch.float32).to(device)
    z = syn_tensor[:, :z_dim]
    w = syn_tensor[:, z_dim : z_dim + w_dim]
    norm_input = pre_decoder(z, w)
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    return syn_num, syn_cat


def recover_data(syn_num, syn_cat, info):

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
    return syn_df


def generate_equiangular_unit_vectors(n, d=None):
    """
    在 d 维单位球面上生成 n 个尽量分散的方向。

    当 d >= n - 1 时，使用正单纯形构造，任意两点夹角相等。
    当 d < n - 1 时，退化为确定性候选采样 + 最远点贪心筛选。
    
    参数:
    n: 生成向量的数量
    d: 维数。如果不指定，默认为 n-1
    
    返回:
    X: 形状为 (n, d) 的阵列，每一行是一个单位向量
    """
    if d is None:
        d = n - 1
    if n <= 0 or d <= 0:
        raise ValueError(f"n 和 d 必须为正数（当前 n={n}, d={d}）")
    if n == 1:
        x = np.zeros((1, d), dtype=np.float64)
        x[0, 0] = 1.0
        return x

    if d < n - 1:
        rng = np.random.default_rng(42)
        candidates = rng.normal(size=(max(512, n * 64), d))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-12

        selected = [0]
        while len(selected) < n:
            chosen = candidates[selected]
            cos_sim = candidates @ chosen.T
            min_distance = (1.0 - cos_sim).min(axis=1)
            min_distance[selected] = -np.inf
            selected.append(int(np.argmax(min_distance)))
        return candidates[selected]

    # Step 1: 构造理论 Gram 矩阵 G
    # 对于 n 个等角向量，其两两之间的余弦值应为 -1/(n-1)
    G = np.full((n, n), -1.0 / (n - 1), dtype=np.float64)
    np.fill_diagonal(G, 1.0)

    # Step 2: 对 Gram 矩阵进行特征分解
    eigenvals, eigenvecs = np.linalg.eigh(G)

    # Step 3: 取出秩为 n-1 的正特征空间，先构造 n-1 维正单纯形嵌入
    positive = np.clip(eigenvals[1:], a_min=0.0, a_max=None)
    X = eigenvecs[:, 1:] * np.sqrt(positive)

    # Step 4: 扩展到目标维度 d
    if d > n - 1:
        X = np.pad(X, ((0, 0), (0, d - (n - 1))), mode="constant")
    elif d < n - 1:
        X = X[:, :d]

    # Step 5: 归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms
