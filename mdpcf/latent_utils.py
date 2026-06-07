import os
import json
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess, get_name_form_args
from mdpcf.csvae.model import Decoder_model


def load_test_latent_arrays(csvae_ckpt_dir):
    test_z = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "test_z.npy"))).float()
    test_w = torch.tensor(np.load(os.path.join(csvae_ckpt_dir, "test_w.npy"))).float()
    return test_z, test_w


def get_input_generate(args):
    dataname = args.dataname
    num_encoding = args.num_encoding

    # activations
    sigmoid = args.sigmoid
    gumbel_softmax = args.gumbel_softmax
    tau = args.tau

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
        test_z, test_w = load_test_latent_arrays(csvae_ckpt_dir)

        pre_decoder = Decoder_model(
            2,
            d_numerical,
            categories,
            4,
            n_head=1,
            factor=32,
            gumbel_softmax=gumbel_softmax,
            sigmoid=sigmoid,
            tau=tau,
        )

        pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location="cpu"))

        info["pre_decoder"] = pre_decoder
        info["latent_dim"] = test_z.shape[-1]
        info["w_dim"] = test_w.shape[-1]

        return {
            "test_z": test_z,
            "test_w": test_w,
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
    syn_tensor = torch.tensor(syn_data, dtype=torch.float32).to(device)
    z = syn_tensor[:, :z_dim]
    w = syn_tensor[:, z_dim:]
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
