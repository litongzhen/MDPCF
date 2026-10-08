import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import csv
import warnings

import os
import shutil
from tqdm import tqdm
import json
import time

from mdpcf.csvae.model import (
    Model_CSVAE,
    Model_StandardVAE,
    Model_ConditionalVAE,
    Encoder_model,
    Decoder_model,
    StandardVAEEncoderModel,
    StandardVAEDecoderModel,
    ConditionalVAEEncoderModel,
    ConditionalVAEDecoderModel,
    BBMLPCLF,
)

from utils_train import (
    preprocess,
    TabularDataset,
    get_name_form_args,
    get_black_box_model_path,
    save_run_config,
)
warnings.filterwarnings("ignore")
import torch.optim as optim

# Transformer 全局超参数
LR = 1e-3  # 学习率
WD = 0  # 权重衰减
D_TOKEN = 4  # Token 嵌入维度 (Transformer 的 d_model 基础)
LATENT_DIM = 4
W_DIM = 2
N_HEAD = 1  # Attention 头数
FACTOR = 32  # 隐藏层维度的倍数因子
NUM_LAYERS = 2  # Transformer 层数
SOURCE_LABEL_MODE = "black_box_prediction"
LATENT_ARTIFACT_PROTOCOL = "cruds_aligned_v3"
CONDITIONAL_VAE_ARTIFACT_PROTOCOL = "conditional_vae_v1"
LATENT_ARTIFACT_METADATA = "latent_artifact_protocol.json"


def checkpoint_artifacts_ready(
    ckpt_dir,
    latent_model="csvae",
    w_structure="global",
    w_dim=W_DIM,
    token_dim=D_TOKEN,
    latent_dim=LATENT_DIM,
):
    """Return whether a checkpoint has all artifacts needed to refresh latent arrays."""
    required_files = [
        "model.pt",
        "encoder.pt",
        "decoder.pt",
        "train_z.npy",
        "test_z.npy",
        LATENT_ARTIFACT_METADATA,
    ]
    if latent_model == "csvae":
        required_files.extend(["train_w.npy", "test_w.npy"])
    if not all(os.path.isfile(os.path.join(ckpt_dir, name)) for name in required_files):
        return False
    try:
        with open(os.path.join(ckpt_dir, LATENT_ARTIFACT_METADATA), "r") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == latent_artifact_metadata(
        latent_model,
        w_structure=w_structure,
        w_dim=w_dim,
        token_dim=token_dim,
        latent_dim=latent_dim,
    )


def encode_label_independent_latents(encoder, x_num, x_cat):
    return encoder(x_num, x_cat, y=None, deterministic=True)[2]


def encode_label_related_latents(encoder, x_num, x_cat, labels):
    return encoder(x_num, x_cat, y=labels, deterministic=True)[5]


def encode_conditional_latents(encoder, x_num, x_cat, labels):
    return encoder(x_num, x_cat, labels, deterministic=True)[2]


def latent_artifact_metadata(
    latent_model,
    w_structure="global",
    w_dim=W_DIM,
    token_dim=D_TOKEN,
    latent_dim=LATENT_DIM,
):
    if latent_model == "standard_vae":
        metadata = {
            "latent_artifact_protocol": "standard_vae_v1",
            "source_label_mode": "not_applicable",
            "search_latent": "z",
            "fixed_latent": "none",
        }
    elif latent_model == "cvae":
        metadata = {
            "latent_artifact_protocol": CONDITIONAL_VAE_ARTIFACT_PROTOCOL,
            "source_label_mode": SOURCE_LABEL_MODE,
            "search_latent": "z",
            "fixed_condition": "target_class",
        }
    elif latent_model == "csvae":
        metadata = {
            "latent_artifact_protocol": LATENT_ARTIFACT_PROTOCOL,
            "source_label_mode": SOURCE_LABEL_MODE,
            "search_latent": "w",
            "fixed_latent": "z",
            "w_structure": w_structure,
            "w_dim": int(w_dim),
        }
    else:
        raise ValueError(f"Unknown latent_model: {latent_model}")
    if (int(token_dim), int(latent_dim)) != (D_TOKEN, LATENT_DIM):
        metadata.update(
            {"token_dim": int(token_dim), "latent_dim": int(latent_dim)}
        )
    return metadata


@torch.no_grad()
def predict_black_box_labels(args, data_dir, info, device):
    encoded, _, _, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )
    train_input = torch.tensor(encoded[0], dtype=torch.float32, device=device)
    test_input = torch.tensor(encoded[1], dtype=torch.float32, device=device)
    classifier = BBMLPCLF(train_input.shape[-1]).to(device)
    classifier.load_state_dict(
        torch.load(get_black_box_model_path(args, data_dir), map_location=device)
    )
    classifier.eval()
    return (
        classifier(train_input).argmax(dim=1),
        classifier(test_input).argmax(dim=1),
    )


def write_latent_artifact_metadata(
    ckpt_dir,
    latent_model,
    w_structure="global",
    w_dim=W_DIM,
    token_dim=D_TOKEN,
    latent_dim=LATENT_DIM,
):
    metadata = latent_artifact_metadata(
        latent_model,
        w_structure=w_structure,
        w_dim=w_dim,
        token_dim=token_dim,
        latent_dim=latent_dim,
    )
    with open(os.path.join(ckpt_dir, LATENT_ARTIFACT_METADATA), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def save_class_prior_artifacts(ckpt_dir, model, latent_model):
    if latent_model != "csvae":
        return
    np.save(
        os.path.join(ckpt_dir, "w_prior_mu.npy"),
        model.w_prior_mu.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(ckpt_dir, "w_prior_logvar.npy"),
        model.w_prior_logvar.detach().cpu().numpy(),
    )


def validation_objective(
    latent_model,
    mse_loss,
    ce_loss,
    kl_z_loss,
    kl_w_loss,
    y_loss,
    beta,
    w_kl_multiplier=1.0,
    adversary_training="joint_grl",
    adversary_confusion=None,
    adversary_weight=1.0,
):
    """Mirror the main-model objective used for checkpoint selection."""
    if latent_model in {"standard_vae", "cvae"}:
        return mse_loss + ce_loss + beta * kl_z_loss
    if adversary_training == "crud_alternating":
        if adversary_confusion is None:
            raise ValueError("CRUD alternating validation requires confusion loss.")
        return (
            mse_loss
            + ce_loss
            + adversary_weight * adversary_confusion
            + beta * kl_z_loss
            + beta * w_kl_multiplier * kl_w_loss
        )
    return (
        mse_loss
        + ce_loss
        + y_loss
        + beta * kl_z_loss
        + beta * w_kl_multiplier * kl_w_loss
    )


def beta_for_epoch(strategy, epoch, target_beta, warmup_epochs, adaptive_beta):
    if strategy == "adaptive":
        return adaptive_beta
    if strategy == "fixed":
        return target_beta
    if strategy == "linear_warmup":
        if warmup_epochs <= 1 or epoch >= warmup_epochs - 1:
            return target_beta
        return target_beta * epoch / (warmup_epochs - 1)
    raise ValueError(f"Unknown beta schedule: {strategy}")


def split_training_validation_indices(size, validation_fraction, seed):
    if size < 2:
        raise ValueError("At least two rows are required for a validation split.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(size)
    validation_size = min(size - 1, max(1, int(round(size * validation_fraction))))
    return shuffled[validation_size:], shuffled[:validation_size]


def adversary_class_weights(labels, num_classes=2, device=None):
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels.reshape(-1), minlength=num_classes).float()
    if counts.numel() != num_classes or torch.any(counts == 0):
        raise ValueError("Training labels must contain all adversary classes.")
    weights = labels.numel() / (num_classes * counts)
    return weights.to(device=device) if device is not None else weights


def adversary_confusion_loss(logits):
    probabilities = torch.softmax(logits, dim=-1)
    return torch.mean(
        torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1)
    )


def z_adversary_logits(model, z, normalize=False):
    adversary_input = z.flatten(start_dim=1)
    if normalize:
        adversary_input = nn.functional.layer_norm(
            adversary_input, (adversary_input.shape[-1],)
        )
    return model.z_adversary(adversary_input)


def conditional_gaussian_kl(mu, logvar, prior_mu, prior_logvar):
    """KL between diagonal Gaussian posteriors and label-conditioned priors."""
    variance_ratio = torch.exp(logvar - prior_logvar)
    mean_distance = (mu - prior_mu).pow(2) * torch.exp(-prior_logvar)
    return 0.5 * torch.mean(
        prior_logvar - logvar + variance_ratio + mean_distance - 1.0
    )


# 损失函数计算
def compute_loss(
    X_num,
    X_cat,
    Recon_X_num,
    Recon_X_cat,
    mu_z,
    logvar_z,
    mu_w,
    logvar_w,
    y_logits,
    y_true,
    w_prior_mu=None,
    w_prior_logvar=None,
    adversary_weights=None,
    gumbel_softmax=False,
    num_reconstr_loss="L2",
    device="cpu",
):
    # === 1. 数值特征重构损失 ===
    # 如果 X_num 是空的张量，则数值损失为零
    if X_num.shape[-1] == 0:
        num_loss = torch.tensor([0]).to(device)
    else:
        # 支持 L2 (MSE) 或 L1 损失
        if num_reconstr_loss == "L2":
            num_loss = (X_num - Recon_X_num).pow(2).mean()
        else:
            num_loss = (X_num - Recon_X_num).abs().mean()

    cat_loss = 0
    acc = 0
    total_num = 0

    if gumbel_softmax:

        cat_loss_fn = nn.L1Loss()

        # cat_loss_fn = nn.CrossEntropyLoss()
        # cat_loss_fn = nn.NLLLoss()

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                # log_x_cat = torch.log(x_cat + 1e-9)
                # cat_loss += cat_loss_fn(log_x_cat, X_cat[:, idx])
                # x_hat = log_x_cat
                # print(x_cat[0])
                # print(X_cat[0, idx])
                # exit(1)
                # 将真实标签转为 One-Hot
                x_cat_ohe = torch.nn.functional.one_hot(
                    X_cat[:, idx], num_classes=x_cat.shape[-1]
                ).float()
                # 计算 Gumbel 输出 (软分布) 与 真实 One-Hot 之间的距离
                cat_loss += cat_loss_fn(x_cat, x_cat_ohe)
                # cat_loss += cat_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat
                x_hat_for_acc = x_hat.argmax(dim=-1)  # 找概率最大的那个类别作为预测结果
                acc += (x_hat_for_acc == X_cat[:, idx]).float().sum()
                total_num += x_hat_for_acc.shape[0]
        cat_loss /= idx + 1
        acc /= total_num
    else:
        cat_loss_fn = nn.CrossEntropyLoss()

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                cat_loss += cat_loss_fn(x_cat, X_cat[:, idx])

                x_hat = x_cat.argmax(dim=-1)
                acc += (x_hat == X_cat[:, idx]).float().sum()
                total_num += x_hat.shape[0]
        cat_loss /= idx + 1
        acc /= total_num
    # loss = num_loss + cat_loss

    temp_z = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld_z = -0.5 * torch.mean(temp_z.mean(-1).mean())
    if w_prior_mu is None or w_prior_logvar is None:
        temp_w = 1 + logvar_w - mu_w.pow(2) - logvar_w.exp()
        loss_kld_w = -0.5 * torch.mean(temp_w.mean(-1).mean())
    else:
        loss_kld_w = conditional_gaussian_kl(
            mu_w, logvar_w, w_prior_mu, w_prior_logvar
        )
    loss_y = nn.CrossEntropyLoss(weight=adversary_weights)(y_logits, y_true.long())
    return num_loss, cat_loss, loss_kld_z, loss_kld_w, loss_y, acc


def forward_latent_model(
    model,
    batch_num,
    batch_cat,
    batch_y,
    latent_model,
    deterministic=False,
    adversary_weight=1.0,
):
    if latent_model == "csvae":
        return model(
            batch_num,
            batch_cat,
            batch_y,
            deterministic=deterministic,
            adversary_weight=adversary_weight,
        )

    if latent_model == "cvae":
        recon_x_num, recon_x_cat, mu_z, logvar_z, z = model(
            batch_num, batch_cat, batch_y, deterministic=deterministic
        )
        w = torch.zeros_like(z)
        mu_w = torch.zeros_like(z)
        logvar_w = torch.zeros_like(z)
        y_logits = torch.zeros(
            batch_num.shape[0], 2, dtype=batch_num.dtype, device=batch_num.device
        )
        return recon_x_num, recon_x_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits

    if latent_model != "standard_vae":
        raise ValueError(f"Unknown latent_model: {latent_model}")

    recon_x_num, recon_x_cat, mu_z, logvar_z, z = model(
        batch_num, batch_cat, deterministic=deterministic
    )
    w = torch.zeros_like(z)
    mu_w = torch.zeros_like(z)
    logvar_w = torch.zeros_like(z)
    y_logits = torch.zeros(
        batch_num.shape[0], 2, dtype=batch_num.dtype, device=batch_num.device
    )
    return recon_x_num, recon_x_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits


def train_black_box_clf_original_data(
    info,
    train_loader,
    test_loader=None,
    num_epochs=100,
    data_dir="",
    cat_encoding=None,
    input_shape=None,
    gumbel_softmax=True,
    hidden_dims=None,
):
    # 节省计算资源和时间 —— 如果相同配置的模型已经训练好了，就不再重复训练
    model_filename = get_black_box_model_path(
        data_dir=data_dir,
        cat_encoding=cat_encoding,
        hidden_dims=hidden_dims,
        gumbel_softmax=gumbel_softmax,
    )

    if os.path.isfile(model_filename):
        print("\tBlack box CLF on original data is already trained")
        # for epoch in range(num_epochs):
        #     pbar = tqdm(train_loader, total=len(train_loader))
        #     pbar.set_description(f"Black Box Epoch {epoch+1}/{num_epochs}")

        #     for batch_num, batch_cat, batch_y in pbar:

        #         batch = torch.cat((batch_num, batch_cat), dim=1)

        #         print(batch[0])
        #         exit(1)
        return
    # 输入维度计算
    input_shape_original = len(info["num_col_idx"]) + len(info["cat_col_idx"])

    if cat_encoding == "one-hot-drop-bin":
        print(
            "Training DIFFERENTIABLE black box CLF on original data (one hot drop binary)..."
        )
    else:
        print("Training DIFFERENTIABLE black box CLF on original data (one hot)...")
    print("Original input shape is", input_shape_original)

    if input_shape is not None:
        print("Given input shape is", input_shape)
    else:
        input_shape = input_shape_original

    # class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}

    # y_n = np.array([class_mapping[class_name[0]] for class_name in y_train])

    # y_train_t = torch.tensor(y_n)

    # 初始化黑盒模型（MLP分类器），input_shape 是特征总数
    clf = BBMLPCLF(input_shape)

    # 定义损失函数：交叉熵损失 (Cross Entropy Loss)
    # 它是分类任务的标准配置，用于衡量预测概率分布与真实标签的差异
    criterion = nn.CrossEntropyLoss()

    # 定义优化器：Adam
    # 它会根据计算出的梯度来更新模型的权重参数，lr=0.001 是学习率
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    # 开始 Epoch 循环：整个数据集要被学习 num_epochs 次
    for epoch in range(num_epochs):

        # 使用 tqdm 创建一个进度条，方便实时看到训练进度
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Black Box Epoch {epoch+1}/{num_epochs}")

        # 从 DataLoader 中取出一批数据（Batch）
        # batch_num: 数值特征部分
        # batch_cat: 类别特征部分
        # batch_y: 真实标签
        for batch_num, batch_cat, batch_y in pbar:
            # 根据预处理方式的不同，决定如何组装输入数据
            if cat_encoding == "one-hot" or cat_encoding == "one-hot-drop-bin":
                # 如果是 One-Hot 模式，batch_num 变量里其实已经包含了所有特征（数值+类别）
                batch = torch.tensor(batch_num)
            else:
                # 如果特征是分开的，需要把数值特征和类别特征在列维度（dim=1）拼接起来
                # 拼成一个 [Batch_Size, Total_Features] 的长向量
                batch = torch.cat((batch_num, batch_cat), dim=1)

            # 开启训练模式
            clf.train()
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播 (Forward Pass)
            outputs = clf(batch)
            # 确保标签是长整型（CrossEntropyLoss 要求标签是整数索引）
            batch_y = batch_y.long()
            # 计算损失
            loss = criterion(outputs, batch_y)
            # 反向传播 (Backward Pass)
            loss.backward()
            # 更新参数
            optimizer.step()

    if test_loader is not None:
        # 切换到评估模式 (Evaluation Mode)
        clf.eval()

        # 初始化计数器，用于计算准确率
        correct = 0
        total = 0
        # 禁用梯度计算
        with torch.no_grad():
            # 遍历测试集数据
            for batch_num, batch_cat, batch_y in test_loader:
                # 保证测试时的输入格式和训练时完全一致
                if cat_encoding == "one-hot" or cat_encoding == "one-hot-drop-bin":
                    batch = torch.tensor(batch_num)
                else:
                    batch = torch.cat((batch_num, batch_cat), dim=1)
                # 前向传播
                outputs = clf(batch)
                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                # 更新总样本数和正确预测数
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total

        print(f"\tInfo: Black box MLP CLF (input) accuracy on test set: {accuracy:.4f}")
    # 保存训练好的模型到指定路径
    torch.save(clf.state_dict(), model_filename)


def main(args):
    args.source_label_mode = SOURCE_LABEL_MODE
    # 读取参数
    dataname = args.dataname
    data_dir = f"data/{dataname}"

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd
    latent_model = getattr(args, "latent_model", "csvae")
    csvae_backbone = getattr(args, "csvae_backbone", "transformer")
    mlp_parameter_match = getattr(args, "mlp_parameter_match", False)
    w_dim = getattr(args, "w_dim", W_DIM)
    token_dim = int(getattr(args, "token_dim", D_TOKEN))
    latent_dim = int(getattr(args, "latent_dim", LATENT_DIM))
    if token_dim <= 0 or latent_dim <= 0:
        raise ValueError("token_dim and latent_dim must be positive")
    if token_dim != latent_dim:
        raise ValueError("latent_dim must equal token_dim for token decoder wrappers")
    w_structure = getattr(args, "w_structure", "global")
    w_kl_multiplier = getattr(args, "w_kl_multiplier", 1.0)

    # encoding
    num_encoding = args.num_encoding

    # loss
    num_reconstr_loss = args.num_reconstr_loss

    # activations
    sigmoid = args.sigmoid
    gumbel_softmax = args.gumbel_softmax
    tau = args.tau
    kl_weight = args.kl_weight
    hidden_dims = args.hidden_dims
    # 创建保存模型权重的文件夹 (ckpt/dataname/实验名)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_name_csvae = get_name_form_args(args)
    ckpt_dir = os.path.join(curr_dir, "ckpt", dataname, output_file_name_csvae)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        print(f"{latent_model.upper()} for setting", output_file_name_csvae, "already exists")
        # exit(0)
    model_save_path = os.path.join(ckpt_dir, "model.pt")
    encoder_save_path = os.path.join(ckpt_dir, "encoder.pt")
    decoder_save_path = os.path.join(ckpt_dir, "decoder.pt")
    save_run_config(args, ckpt_dir)

    device = args.device
    batch_size = 4096
    info_path = os.path.join("data", dataname, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    # 调用 preprocess，使用 args.num_encoding 指定的编码方式
    X_num, X_cat, categories, d_numerical, y = preprocess(
        data_dir, task_type=info["task_type"], num_encoding=num_encoding
    )

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat
    # 转成 Tensor
    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    y_train, y_test = y

    full_train_data = TabularDataset(
        X_train_num.float(), X_cat=X_train_cat, y=y_train, info=info
    )
    validation_fraction = getattr(args, "validation_fraction", 0.0)
    if validation_fraction > 0:
        train_indices, validation_indices = split_training_validation_indices(
            len(X_train_num), validation_fraction, args.seed
        )
        train_data = Subset(full_train_data, train_indices.tolist())
        X_validation_num = X_train_num[validation_indices].float().to(device)
        X_validation_cat = X_train_cat[validation_indices].to(device)
        y_validation = full_train_data.y[validation_indices].to(device)
    else:
        train_data = full_train_data
        X_validation_num = X_test_num.float().to(device)
        X_validation_cat = X_test_cat.to(device)
        y_validation = TabularDataset(
            X_test_num.float(), X_cat=X_test_cat, y=y_test, info=info
        ).y.to(device)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    adversary_weights = None
    if latent_model == "csvae" and getattr(args, "z_adversary_class_balance", False):
        adversary_labels = (
            full_train_data.y[train_indices]
            if validation_fraction > 0
            else full_train_data.y
        )
        adversary_weights = adversary_class_weights(
            adversary_labels, num_classes=2, device=device
        )

    black_box_path = get_black_box_model_path(args, data_dir)

    if not os.path.isfile(black_box_path):
        raise FileNotFoundError(
            f"找不到黑盒分类器: {black_box_path}。请先运行 `--method black_box --mode train/run`。"
        )
    print(f"Found black box classifier: {black_box_path}")
    train_source_labels, test_source_labels = predict_black_box_labels(
        args, data_dir, info, device
    )

    if latent_model == "standard_vae":
        model = Model_StandardVAE(
            NUM_LAYERS, d_numerical, categories, token_dim,
            dim_z=latent_dim, n_head=N_HEAD, factor=FACTOR, bias=True,
            gumbel_softmax=gumbel_softmax, tau=tau, sigmoid=sigmoid,
            backbone=csvae_backbone,
        )
        pre_encoder = StandardVAEEncoderModel(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, backbone=csvae_backbone,
        ).to(device)
        pre_decoder = StandardVAEDecoderModel(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, gumbel_softmax=gumbel_softmax, sigmoid=sigmoid,
            backbone=csvae_backbone,
        ).to(device)
    elif latent_model == "cvae":
        model = Model_ConditionalVAE(
            NUM_LAYERS, d_numerical, categories, token_dim,
            dim_z=latent_dim, n_head=N_HEAD, factor=FACTOR, bias=True,
            gumbel_softmax=gumbel_softmax, tau=tau, sigmoid=sigmoid,
            backbone=csvae_backbone,
        )
        pre_encoder = ConditionalVAEEncoderModel(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, backbone=csvae_backbone,
        ).to(device)
        pre_decoder = ConditionalVAEDecoderModel(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, gumbel_softmax=gumbel_softmax, sigmoid=sigmoid,
            backbone=csvae_backbone,
        ).to(device)
    elif latent_model == "csvae":
        model = Model_CSVAE(
            NUM_LAYERS,
            d_numerical,
            categories,
            token_dim,
            dim_z=latent_dim,
            dim_w=w_dim,
            n_head=N_HEAD,
            factor=FACTOR,
            bias=True,
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
            backbone=csvae_backbone,
            w_structure=w_structure,
            mlp_parameter_match=mlp_parameter_match,
        )
        pre_encoder = Encoder_model(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, backbone=csvae_backbone, dim_w=w_dim,
            mlp_parameter_match=mlp_parameter_match,
        ).to(device)
        pre_decoder = Decoder_model(
            NUM_LAYERS, d_numerical, categories, token_dim, n_head=N_HEAD,
            factor=FACTOR, gumbel_softmax=gumbel_softmax, sigmoid=sigmoid,
            backbone=csvae_backbone, dim_w=w_dim,
            mlp_parameter_match=mlp_parameter_match,
        ).to(device)
    else:
        raise ValueError(f"Unknown latent_model: {latent_model}")
    model = model.to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    if getattr(args, "reuse_checkpoint", False) and checkpoint_artifacts_ready(
        ckpt_dir,
        latent_model,
        w_structure=w_structure,
        w_dim=w_dim,
        token_dim=token_dim,
        latent_dim=latent_dim,
    ):
        print(f"Reusing complete {latent_model.upper()} checkpoint and refreshing latent arrays: {ckpt_dir}")
        pre_encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
        pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))
        with torch.no_grad():
            X_train_num = X_train_num.to(device)
            X_train_cat = X_train_cat.to(device)
            if latent_model == "csvae":
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                model.eval()
                train_z = encode_label_independent_latents(
                    model.encoder, X_train_num, X_train_cat
                ).detach().cpu().numpy()
                test_z = encode_label_independent_latents(
                    model.encoder, X_test_num, X_test_cat
                ).detach().cpu().numpy()
                train_w = encode_label_related_latents(
                    model.encoder, X_train_num, X_train_cat, train_source_labels
                ).detach().cpu().numpy()
                test_w = encode_label_related_latents(
                    model.encoder, X_test_num, X_test_cat, test_source_labels
                ).detach().cpu().numpy()
            elif latent_model == "cvae":
                train_z = encode_conditional_latents(
                    pre_encoder.model, X_train_num, X_train_cat, train_source_labels
                ).detach().cpu().numpy()
                test_z = encode_conditional_latents(
                    pre_encoder.model, X_test_num, X_test_cat, test_source_labels
                ).detach().cpu().numpy()
            else:
                train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
                test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
            np.save(f"{ckpt_dir}/train_z.npy", train_z)
            np.save(f"{ckpt_dir}/test_z.npy", test_z)
            if latent_model == "csvae":
                np.save(f"{ckpt_dir}/train_w.npy", train_w)
                np.save(f"{ckpt_dir}/test_w.npy", test_w)
                save_class_prior_artifacts(ckpt_dir, model, latent_model)
            write_latent_artifact_metadata(
                ckpt_dir,
                latent_model,
                w_structure=w_structure,
                w_dim=w_dim,
                token_dim=token_dim,
                latent_dim=latent_dim,
            )
        return

    adversary_training = getattr(args, "z_adversary_training", "joint_grl")
    adversary_optimizer = None
    if latent_model == "csvae" and adversary_training == "crud_alternating":
        adversary_parameter_ids = {
            id(parameter) for parameter in model.z_adversary.parameters()
        }
        main_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in adversary_parameter_ids
        ]
        optimizer = torch.optim.Adam(main_parameters, lr=LR, weight_decay=WD)
        adversary_optimizer = torch.optim.Adam(
            model.z_adversary.parameters(), lr=LR, weight_decay=WD
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_lr,
    )

    num_epochs = args.csvae_epochs
    best_train_loss = float("inf")
    best_epoch = -1
    history_path = os.path.join(ckpt_dir, "training_history.csv")
    history_fields = [
        "epoch",
        "beta",
        "beta_w",
        "learning_rate",
        "train_mse",
        "train_ce",
        "train_kl",
        "val_mse",
        "val_ce",
        "val_kl_z",
        "val_kl_w",
        "val_y",
        "val_y_confusion",
        "val_acc",
        "val_objective",
        "is_best",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as history_file:
        csv.DictWriter(history_file, fieldnames=history_fields).writeheader()

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    adaptive_beta = max_beta

    if kl_weight > 0:
        adaptive_beta = 1 / num_epochs

    start_time = time.time()
    for epoch in range(num_epochs):
        beta = beta_for_epoch(
            args.beta_schedule,
            epoch,
            args.beta_target,
            args.beta_warmup_epochs,
            adaptive_beta,
        )
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        # if kl_weight > 0:
        #     beta = epoch * kl_weight / num_epochs

        for batch_num, batch_cat, batch_y in pbar:
            model.train()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            if adversary_optimizer is not None:
                with torch.no_grad():
                    adversary_z = model.encoder(
                        batch_num,
                        batch_cat,
                        batch_y,
                        adversary_weight=0.0,
                    )[2]
                for _ in range(args.z_adversary_steps):
                    adversary_optimizer.zero_grad()
                    adversary_logits = z_adversary_logits(
                        model,
                        adversary_z.detach(),
                        normalize=args.z_adversary_normalize,
                    )
                    adversary_classifier_loss = nn.CrossEntropyLoss(
                        weight=adversary_weights
                    )(adversary_logits, batch_y.long())
                    adversary_classifier_loss.backward()
                    adversary_optimizer.step()
                for parameter in model.z_adversary.parameters():
                    parameter.requires_grad_(False)

            optimizer.zero_grad()

            (
                Recon_X_num,
                Recon_X_cat,
                mu_z,
                logvar_z,
                z,
                mu_w,
                logvar_w,
                w,
                y_logits,
            ) = forward_latent_model(
                model,
                batch_num,
                batch_cat,
                batch_y,
                latent_model,
                adversary_weight=(
                    0.0 if adversary_optimizer is not None else args.z_adversary_weight
                ),
            )

            if latent_model == "csvae":
                w_prior_mu, w_prior_logvar = model.class_prior(
                    batch_y,
                    n_tokens=mu_w.shape[1] if mu_w.dim() == 3 else None,
                )
            else:
                w_prior_mu = None
                w_prior_logvar = None

            loss_mse, loss_ce, loss_kld_z, loss_kld_w, loss_y, train_acc = compute_loss(
                batch_num,
                batch_cat,
                Recon_X_num,
                Recon_X_cat,
                mu_z,
                logvar_z,
                mu_w,
                logvar_w,
                y_logits,
                batch_y,
                w_prior_mu=w_prior_mu,
                w_prior_logvar=w_prior_logvar,
                adversary_weights=adversary_weights,
                gumbel_softmax=gumbel_softmax,
                num_reconstr_loss=num_reconstr_loss,
                device=device,
            )

            if latent_model in {"standard_vae", "cvae"}:
                loss = loss_mse + loss_ce + beta * loss_kld_z
                loss_kld_w = torch.tensor(0.0, device=device)
                loss_y = torch.tensor(0.0, device=device)
            else:
                adversary_main_loss = (
                    args.z_adversary_weight
                    * adversary_confusion_loss(
                        z_adversary_logits(
                            model,
                            z,
                            normalize=args.z_adversary_normalize,
                        )
                    )
                    if adversary_optimizer is not None
                    else loss_y
                )
                loss = (
                    loss_mse
                    + loss_ce
                    + adversary_main_loss
                    + beta * loss_kld_z
                    + beta * w_kl_multiplier * loss_kld_w
                )
            loss.backward()
            optimizer.step()
            if adversary_optimizer is not None:
                for parameter in model.z_adversary.parameters():
                    parameter.requires_grad_(True)

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += (loss_kld_z.item() + loss_kld_w.item()) * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        """
            Evaluation
        """
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits = forward_latent_model(
                model,
                X_validation_num,
                X_validation_cat,
                y_validation,
                latent_model,
                deterministic=True,
                adversary_weight=(
                    0.0 if adversary_optimizer is not None else args.z_adversary_weight
                ),
            )
            if latent_model == "csvae":
                val_prior_mu, val_prior_logvar = model.class_prior(
                    y_validation,
                    n_tokens=mu_w.shape[1] if mu_w.dim() == 3 else None,
                )
            else:
                val_prior_mu = None
                val_prior_logvar = None

            val_mse_loss, val_ce_loss, val_kl_z, val_kl_w, val_y_loss, val_acc = compute_loss(
                X_validation_num,
                X_validation_cat,
                Recon_X_num,
                Recon_X_cat,
                mu_z,
                logvar_z,
                mu_w,
                logvar_w,
                y_logits,
                y_validation,
                w_prior_mu=val_prior_mu,
                w_prior_logvar=val_prior_logvar,
                adversary_weights=adversary_weights,
                gumbel_softmax=gumbel_softmax,
            )
            if latent_model in {"standard_vae", "cvae"}:
                val_kl_w = torch.tensor(0.0, device=device)
                val_y_loss = torch.tensor(0.0, device=device)
                val_y_confusion = torch.tensor(0.0, device=device)
            else:
                val_y_confusion = adversary_confusion_loss(
                    z_adversary_logits(
                        model,
                        z,
                        normalize=args.z_adversary_normalize,
                    )
                )
            epoch_beta = beta
            val_loss = validation_objective(
                latent_model,
                val_mse_loss,
                val_ce_loss,
                val_kl_z,
                val_kl_w,
                val_y_loss,
                epoch_beta,
                w_kl_multiplier=w_kl_multiplier,
                adversary_training=adversary_training,
                adversary_confusion=val_y_confusion,
                adversary_weight=args.z_adversary_weight,
            ).item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            is_best = val_loss < best_train_loss
            if is_best:
                best_train_loss = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if args.beta_schedule == "adaptive":
                        if kl_weight < 0:
                            if adaptive_beta > min_beta:
                                adaptive_beta = adaptive_beta * lambd
                        elif adaptive_beta < kl_weight:
                            adaptive_beta = adaptive_beta * 1.3

            with open(history_path, "a", newline="", encoding="utf-8") as history_file:
                csv.DictWriter(history_file, fieldnames=history_fields).writerow(
                    {
                        "epoch": epoch,
                        "beta": epoch_beta,
                        "beta_w": epoch_beta * w_kl_multiplier,
                        "learning_rate": current_lr,
                        "train_mse": num_loss,
                        "train_ce": cat_loss,
                        "train_kl": kl_loss,
                        "val_mse": val_mse_loss.item(),
                        "val_ce": val_ce_loss.item(),
                        "val_kl_z": val_kl_z.item(),
                        "val_kl_w": val_kl_w.item(),
                        "val_y": val_y_loss.item(),
                        "val_y_confusion": val_y_confusion.item(),
                        "val_acc": val_acc.item(),
                        "val_objective": val_loss,
                        "is_best": int(is_best),
                    }
                )

            completed_epochs = epoch + 1
            if completed_epochs in {1000, 2000, num_epochs}:
                milestone_path = os.path.join(
                    ckpt_dir, f"model_best_through_epoch_{completed_epochs}.pt"
                )
                shutil.copyfile(model_save_path, milestone_path)

        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print(
                "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                    epoch,
                    beta,
                    num_loss,
                    cat_loss,
                    kl_loss,
                val_mse_loss.item(),
                val_ce_loss.item(),
                train_acc.item(),
                val_acc.item(),
            )
        )

    # Define the file path where you want to save the log
    log_file_path = f"{ckpt_dir}/final_log.txt"

    # Open the file in append mode to add new logs at the end of the file
    with open(log_file_path, "a") as log_file:
        log_file.write(
                "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}\n".format(
                    epoch,
                    beta,
                    num_loss,
                    cat_loss,
                kl_loss,
                val_mse_loss.item(),
                val_ce_loss.item(),
                train_acc.item(),
                val_acc.item(),
            )
        )

    end_time = time.time()
    print("Training time: {:.4f} mins".format((end_time - start_time) / 60))
    print(f"Reloading best checkpoint from epoch {best_epoch + 1}: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Saving latent embeddings
    with torch.no_grad():
        model.eval()
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print("Successfully load and save the model!")

        if latent_model == "csvae":
            train_z = encode_label_independent_latents(
                model.encoder, X_train_num, X_train_cat
            ).detach().cpu().numpy()
            test_z = encode_label_independent_latents(
                model.encoder, X_test_num, X_test_cat
            ).detach().cpu().numpy()
            train_w = encode_label_related_latents(
                model.encoder, X_train_num, X_train_cat, train_source_labels
            ).detach().cpu().numpy()
            test_w = encode_label_related_latents(
                model.encoder, X_test_num, X_test_cat, test_source_labels
            ).detach().cpu().numpy()
        elif latent_model == "cvae":
            train_z = pre_encoder(
                X_train_num, X_train_cat, train_source_labels
            ).detach().cpu().numpy()
            test_z = pre_encoder(
                X_test_num, X_test_cat, test_source_labels
            ).detach().cpu().numpy()
        else:
            train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
            test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
        np.save(f"{ckpt_dir}/train_z.npy", train_z)
        np.save(f"{ckpt_dir}/test_z.npy", test_z)
        if latent_model == "csvae":
            np.save(f"{ckpt_dir}/train_w.npy", train_w)
            np.save(f"{ckpt_dir}/test_w.npy", test_w)
            save_class_prior_artifacts(ckpt_dir, model, latent_model)
        write_latent_artifact_metadata(
            ckpt_dir,
            latent_model,
            w_structure=w_structure,
            w_dim=w_dim,
            token_dim=token_dim,
            latent_dim=latent_dim,
        )

        print("Successfully save pretrained embeddings in disk!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conditional Variational Autoencoder")

    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--max_beta", type=float, default=1e-2, help="Initial Beta.")
    parser.add_argument("--min_beta", type=float, default=1e-5, help="Minimum Beta.")
    parser.add_argument("--lambd", type=float, default=0.7, help="Decay of Beta.")

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
