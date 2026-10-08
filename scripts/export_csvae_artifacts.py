import argparse
import os
import json
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mdpcf.csvae.main import (
    encode_label_independent_latents,
    encode_label_related_latents,
    predict_black_box_labels,
    save_class_prior_artifacts,
    write_latent_artifact_metadata,
)
from mdpcf.csvae.model import Model_CSVAE, Encoder_model, Decoder_model
from utils import get_args
from utils_train import preprocess, get_name_form_args, save_run_config


def parse_args():
    base_args = get_args()
    default_device = getattr(base_args, "device", "cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(
        description="Export CSVAE encoder, decoder, and latent arrays from model.pt."
    )
    parser.add_argument("--dataname", default=base_args.dataname)
    parser.add_argument("--num_encoding", default=base_args.num_encoding)
    parser.add_argument("--num_reconstr_loss", default=base_args.num_reconstr_loss)
    parser.add_argument("--gumbel_softmax", default=base_args.gumbel_softmax)
    parser.add_argument("--sigmoid", default=base_args.sigmoid)
    parser.add_argument("--tau", type=float, default=base_args.tau)
    parser.add_argument("--kl_weight", type=float, default=base_args.kl_weight)
    parser.add_argument(
        "--csvae_backbone",
        choices=["transformer", "mlp"],
        default=getattr(base_args, "csvae_backbone", "transformer"),
    )
    parser.add_argument("--hidden_dims", type=int, default=base_args.hidden_dims)
    parser.add_argument("--seed", type=int, default=base_args.seed)
    parser.add_argument("--black_box_seed", type=int, default=base_args.black_box_seed)
    parser.add_argument("--run_tag", default=base_args.run_tag)
    parser.add_argument("--latent_model", default="csvae", choices=["csvae"])
    parser.add_argument("--w_dim", type=int, default=base_args.w_dim)
    parser.add_argument(
        "--w_structure", choices=["global", "token"], default=base_args.w_structure
    )
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


@torch.no_grad()
def main():
    args = parse_args()
    args.gumbel_softmax = as_bool(args.gumbel_softmax)
    args.sigmoid = as_bool(args.sigmoid)

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    dataname = args.dataname
    data_dir = os.path.join("data", dataname)
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        num_encoding=args.num_encoding,
    )
    X_train_num, X_test_num = [torch.tensor(x).float().to(device) for x in X_num]
    X_train_cat, X_test_cat = [torch.tensor(x).to(device) for x in X_cat]
    train_labels, test_labels = predict_black_box_labels(
        args, data_dir, info, device
    )

    output_name = get_name_form_args(args)
    ckpt_dir = os.path.join("mdpcf", "csvae", "ckpt", dataname, output_name)
    model_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing CSVAE model checkpoint: {model_path}")

    model = Model_CSVAE(
        2,
        d_numerical,
        categories,
        4,
        dim_z=4,
        dim_w=args.w_dim,
        n_head=1,
        factor=32,
        bias=True,
        gumbel_softmax=args.gumbel_softmax,
        tau=args.tau,
        sigmoid=args.sigmoid,
        backbone=args.csvae_backbone,
        w_structure=args.w_structure,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    encoder = Encoder_model(
        2,
        d_numerical,
        categories,
        4,
        n_head=1,
        factor=32,
        backbone=args.csvae_backbone,
        dim_w=args.w_dim,
    ).to(device)
    decoder = Decoder_model(
        2,
        d_numerical,
        categories,
        4,
        n_head=1,
        factor=32,
        gumbel_softmax=args.gumbel_softmax,
        sigmoid=args.sigmoid,
        tau=args.tau,
        backbone=args.csvae_backbone,
        dim_w=args.w_dim,
    ).to(device)
    encoder.load_weights(model)
    decoder.load_weights(model)
    encoder.eval()
    decoder.eval()

    os.makedirs(ckpt_dir, exist_ok=True)
    save_run_config(args, ckpt_dir)
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, "decoder.pt"))

    train_z = encode_label_independent_latents(
        model.encoder, X_train_num, X_train_cat
    ).detach().cpu().numpy()
    test_z = encode_label_independent_latents(
        model.encoder, X_test_num, X_test_cat
    ).detach().cpu().numpy()
    train_w = encode_label_related_latents(
        model.encoder, X_train_num, X_train_cat, train_labels
    ).detach().cpu().numpy()
    test_w = encode_label_related_latents(
        model.encoder, X_test_num, X_test_cat, test_labels
    ).detach().cpu().numpy()
    np.save(os.path.join(ckpt_dir, "train_z.npy"), train_z)
    np.save(os.path.join(ckpt_dir, "test_z.npy"), test_z)
    np.save(os.path.join(ckpt_dir, "train_w.npy"), train_w)
    np.save(os.path.join(ckpt_dir, "test_w.npy"), test_w)
    save_class_prior_artifacts(ckpt_dir, model, "csvae")
    write_latent_artifact_metadata(ckpt_dir, "csvae")

    print(f"Exported CSVAE artifacts to {ckpt_dir}")


if __name__ == "__main__":
    main()
