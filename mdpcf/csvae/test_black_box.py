import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from mdpcf.csvae.model import BBMLPCLF
from utils_train import preprocess, get_black_box_model_path


def main(args):
    dataname = args.dataname
    data_dir = f"data/{dataname}"
    hidden_dims = args.hidden_dims
    gumbel_softmax = args.gumbel_softmax
    num_encoding = args.num_encoding

    with open(f"{data_dir}/info.json", "r") as f:
        info = json.load(f)

    X_num_cat_ohe, _, _, _, y_ohe = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )
    _, X_test = X_num_cat_ohe
    _, y_test_raw = y_ohe

    class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}
    y_test = np.array([class_mapping[row[0]] for row in y_test_raw])

    model_filename = get_black_box_model_path(args, data_dir)

    if not os.path.isfile(model_filename):
        raise FileNotFoundError(
            f"找不到黑盒模型文件: {model_filename}。请先运行 mode=train 训练。"
        )

    device = args.device
    model = BBMLPCLF(X_test.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, probs[:, 1])
    except Exception:
        auc = float("nan")

    print(f"Model: {model_filename}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))
    print(
        classification_report(
            y_test,
            pred,
            target_names=[info["negative_class"].strip(), info["target_class"].strip()],
            digits=4,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-box model test")
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_dims", type=int, default=16)
    parser.add_argument("--gumbel_softmax", type=bool, default=True)
    parser.add_argument("--num_encoding", type=str, default="min_max_torch")
    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    main(args)
