import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time
import pandas as pd

from mdpcf.vae.model import (
    Model_VAE,
    Encoder_model,
    Decoder_model,
    BBMLPCLF,
    Encoder_model_Z,
)

from utils_train import preprocess, TabularDataset, get_name_form_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch.optim as optim

# Transformer 全局超参数
LR = 1e-3  # 学习率
WD = 0  # 权重衰减
D_TOKEN = 4  # Token 嵌入维度 (Transformer 的 d_model 基础)
LATENT_DIM = 4
W_DIM = 4
TOKEN_BIAS = True
N_HEAD = 1  # Attention 头数
FACTOR = 32  # 隐藏层维度的倍数因子
NUM_LAYERS = 2  # Transformer 层数


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
    temp_w = 1 + logvar_w - mu_w.pow(2) - logvar_w.exp()
    loss_kld_w = -0.5 * torch.mean(temp_w.mean(-1).mean())
    loss_y = nn.CrossEntropyLoss()(y_logits, y_true.long())
    return num_loss, cat_loss, loss_kld_z, loss_kld_w, loss_y, acc


def concatenate_input_dims(x, tensor=False):
    shape = x.shape
    # shape[:-2] -> 获取除了最后两维之外的所有维度（通常是 Batch Size）
    # shape[-2] * shape[-1] -> 计算最后两维的乘积（合并后的总特征数）
    new_shape = shape[:-2] + (shape[-2] * shape[-1],)

    # 执行 reshape 操作，把最后两维压扁
    reshaped_x = x.reshape(new_shape)

    # 如果需要，转成 PyTorch Tensor
    if tensor:
        return torch.tensor(reshaped_x)
    return reshaped_x


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
    model_filename = f"{data_dir}/black_box_mlp.pkl"
    if cat_encoding == "one-hot-drop-bin":
        model_filename = f"{data_dir}/black_box_mlp_ohe_drop_bin.pkl"

    if not gumbel_softmax:
        model_filename = f"{data_dir}/black_box_mlp_no_gumbel_softmax.pkl"

    if hidden_dims is not None:
        model_filename = f"{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl"

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


def train_black_box_clf_latent(X, y, info, X_test=None, y_test=None, ckpt_dir=""):
    # X 是 Encoder 输出的潜在向量序列
    # 形状通常是 [Batch_Size, Num_Tokens, Token_Dim] (3D Tensor)

    # 切掉 [CLS] Token
    # Transformer 通常会在序列最前面加一个特殊的 [CLS] token (index 0)。
    # 这里只取 index 1 之后的部分，即具体的特征 token。
    X = X[:, 1:, :]
    # 把最后两个维度 (Num_Tokens, Token_Dim) 压扁成一个长向量，变成 [Batch_Size, Total_Latent_Dim]，这样才能喂给 MLP。
    X = concatenate_input_dims(X, tensor=True)
    # 映射类别标签到数值
    class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}
    y_n = np.array([class_mapping[class_name[0]] for class_name in y])
    y = torch.tensor(y_n)
    # 初始化 MLP 分类器，输入维度 X.shape[-1] 是展平后的潜在向量长度
    clf = BBMLPCLF(X.shape[-1])
    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy Loss for classification
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = clf(X)  # Forward pass
        loss = criterion(outputs, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

    model_filename = f"{ckpt_dir}/black_box_latent.pt"

    torch.save(clf.state_dict(), model_filename)

    if X_test is not None:

        X_test = X_test[:, 1:, :]
        X_test = concatenate_input_dims(X_test, tensor=True)

        y_test_n = np.array([class_mapping[class_name[0]] for class_name in y_test])

        # Put the model in evaluation mode
        clf.eval()

        # Pass the test dataset through the clf to get predictions
        with torch.no_grad():
            test_outputs = clf(X_test)

        _, y_pred = torch.max(test_outputs, 1)
        acc = accuracy_score(y_test_n, y_pred)
        print(f"\tInfo: Black box MLP clf (latent) accuracy on test set: {acc:.4f}")


def main(args):
    # 读取参数
    dataname = args.dataname
    data_dir = f"data/{dataname}"

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    reparam = args.reparam

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
    output_file_name_vae = get_name_form_args(args)
    # 创建保存模型权重的文件夹 (ckpt/dataname/实验名)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(curr_dir, "ckpt", dataname, output_file_name_vae)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        print("CSVAE for setting", output_file_name_vae, "already exists")
        # exit(0)
    model_save_path = os.path.join(ckpt_dir, "model.pt")
    encoder_save_path = os.path.join(ckpt_dir, "encoder.pt")
    decoder_save_path = os.path.join(ckpt_dir, "decoder.pt")

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

    train_data_clf = TabularDataset(
        X_train_num.float(), X_cat=X_train_cat, y=y_train, info=info
    )
    test_data_clf = TabularDataset(
        X_test_num.float(), X_cat=X_test_cat, y=y_test, info=info
    )

    train_data = TabularDataset(
        X_train_num.float(), X_cat=X_train_cat, y=y_train, info=info
    )

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    black_box_path = f"{data_dir}/black_box_mlp.pkl"
    if hidden_dims is not None:
        black_box_path = f"{data_dir}/black_box_mlp_hidden_{hidden_dims}.pkl"

    if not os.path.isfile(black_box_path):
        raise FileNotFoundError(
            f"找不到黑盒分类器: {black_box_path}。请先运行 `--method black_box --mode train/run`。"
        )
    print(f"Found black box classifier: {black_box_path}")

    model = Model_VAE(
        NUM_LAYERS,
        d_numerical,
        categories,
        D_TOKEN,
        dim_z=LATENT_DIM,
        dim_w=W_DIM,
        n_head=N_HEAD,
        factor=FACTOR,
        bias=True,
        gumbel_softmax=gumbel_softmax,
        tau=tau,
        sigmoid=sigmoid,
    )
    model = model.to(device)

    if reparam:
        pre_encoder = Encoder_model_Z(
            NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
        ).to(device)
    else:
        pre_encoder = Encoder_model(
            NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
        ).to(device)

    pre_decoder = Decoder_model(
        NUM_LAYERS,
        d_numerical,
        categories,
        D_TOKEN,
        n_head=N_HEAD,
        factor=FACTOR,
        gumbel_softmax=gumbel_softmax,
        sigmoid=sigmoid,
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.95, patience=10
    )

    num_epochs = args.vae_epochs
    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    beta = max_beta

    if kl_weight > 0:
        beta = 1 / num_epochs

    start_time = time.time()
    for epoch in range(num_epochs):
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
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

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
            ) = model(batch_num, batch_cat, batch_y)

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
                gumbel_softmax=gumbel_softmax,
                num_reconstr_loss=num_reconstr_loss,
                device=device,
            )

            loss = loss_mse + loss_ce + loss_y + beta * (loss_kld_z + loss_kld_w)
            loss.backward()
            optimizer.step()

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
            y_test_eval = test_data_clf.y.to(device)
            Recon_X_num, Recon_X_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits = model(X_test_num, X_test_cat, y=y_test_eval)

            val_mse_loss, val_ce_loss, val_kl_z, val_kl_w, val_y_loss, val_acc = compute_loss(
                X_test_num,
                X_test_cat,
                Recon_X_num,
                Recon_X_cat,
                mu_z,
                logvar_z,
                mu_w,
                logvar_w,
                y_logits,
                y_test_eval,
                gumbel_softmax=gumbel_softmax,
            )
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item() + val_y_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if kl_weight < 0:
                        if beta > min_beta:
                            beta = beta * lambd
                    else:
                        if beta < kl_weight:
                            beta = beta * 1.3

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

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print("Successfully load and save the model!")

        y_train_tensor = train_data_clf.y.to(device)
        y_test_tensor = test_data_clf.y.to(device)
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
        train_w = model.encoder(X_train_num, X_train_cat, y=y_train_tensor)[5].detach().cpu().numpy()
        test_w = model.encoder(X_test_num, X_test_cat, y=y_test_tensor)[5].detach().cpu().numpy()
        np.save(f"{ckpt_dir}/train_z.npy", train_z)
        np.save(f"{ckpt_dir}/test_z.npy", test_z)
        np.save(f"{ckpt_dir}/train_w.npy", train_w)
        np.save(f"{ckpt_dir}/test_w.npy", test_w)

        print("Successfully save pretrained embeddings in disk!")

    # train_black_box_clf_latent(train_z, y_train, info, X_test=test_z, y_test=y_test, ckpt_dir=ckpt_dir)


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
