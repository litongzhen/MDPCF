import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import argparse
import warnings

import os
from tqdm import tqdm
import json

from mdpcf.csvae.model import BBMLPCLF

from utils_train import (
    preprocess,
    TabularDataset,
    get_black_box_model_path,
    save_run_config,
)
warnings.filterwarnings("ignore")
import torch.optim as optim


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
    model_filename=None,
):

    if model_filename is None:
        model_filename = get_black_box_model_path(
            data_dir=data_dir,
            cat_encoding=cat_encoding,
            hidden_dims=hidden_dims,
            gumbel_softmax=gumbel_softmax,
        )

    if os.path.isfile(model_filename):
        print("\tBlack box CLF on original data is already trained")
        return

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

    clf = BBMLPCLF(input_shape)

    criterion = nn.CrossEntropyLoss()  # Use Cross Entropy Loss for classification
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Black Box Epoch {epoch+1}/{num_epochs}")

        for batch_num, batch_cat, batch_y in pbar:

            if cat_encoding == "one-hot" or cat_encoding == "one-hot-drop-bin":
                batch = torch.tensor(batch_num)
            else:
                batch = torch.cat((batch_num, batch_cat), dim=1)

            clf.train()
            optimizer.zero_grad()
            outputs = clf(batch)
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

    if test_loader is not None:
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in test_loader:
                if cat_encoding == "one-hot" or cat_encoding == "one-hot-drop-bin":
                    batch = torch.tensor(batch_num)
                else:
                    batch = torch.cat((batch_num, batch_cat), dim=1)
                outputs = clf(batch)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total

        print(f"\tInfo: Black box MLP CLF (input) accuracy on test set: {accuracy:.4f}")
    torch.save(clf.state_dict(), model_filename)


def main(args):

    dataname = args.dataname
    data_dir = f"data/{dataname}"

    hidden_dims = args.hidden_dims

    # encoding
    num_encoding = args.num_encoding

    gumbel_softmax = args.gumbel_softmax

    device = args.device
    batch_size = 4096
    num_epochs = args.black_box_epochs

    info_path = f"data/{dataname}/info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical, y = preprocess(
        data_dir, task_type=info["task_type"], num_encoding=num_encoding
    )

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    # ----------------------------------------------------------------------------

    X_num_cat_ohe, _, categories_ohe, d_numerical_ohe, y_ohe = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )

    X_train_num_cat_ohe, X_test_num_cat_ohe = X_num_cat_ohe

    y_train_num_cat_ohe, y_test_num_cat_ohe = y_ohe

    train_data_clf_ohe = TabularDataset(
        X_train_num_cat_ohe, y=y_train_num_cat_ohe, info=info
    )

    train_loader_clf_ohe = DataLoader(
        train_data_clf_ohe,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_data_clf_ohe = TabularDataset(
        X_test_num_cat_ohe, y=y_test_num_cat_ohe, info=info
    )

    test_loader_clf_ohe = DataLoader(
        test_data_clf_ohe,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ----------------------------------------------------------------------------

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    model_filename = get_black_box_model_path(args, data_dir, cat_encoding="one-hot")
    save_run_config(
        args,
        data_dir,
        filename=f"{os.path.splitext(os.path.basename(model_filename))[0]}_config.json",
    )

    train_black_box_clf_original_data(
        info,
        train_loader_clf_ohe,
        test_loader=test_loader_clf_ohe,
        data_dir=data_dir,
        num_epochs=num_epochs,
        cat_encoding="one-hot",
        input_shape=X_train_num_cat_ohe.shape[-1],
        gumbel_softmax=gumbel_softmax,
        hidden_dims=hidden_dims,
        model_filename=model_filename,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Variational Autoencoder")

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
