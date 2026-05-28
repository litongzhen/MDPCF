# train.py
import torch
import torch.utils.data
import torch.nn.functional as F
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
from tqdm import trange
from model import EncoderW, EncoderZ, DecoderX, DecoderY, _kl_gaussian  # 使用标准化 KL

# ----------------------------
# Data Generation
# ----------------------------
X_, _ = sklearn.datasets.make_swiss_roll(n_samples=100_000, random_state=207)
X_ = X_.astype(np.float32)
y_ = (X_[:, 1] < 10).astype(np.int32)
print(f'Swiss roll data shape: {X_.shape}, labels shape: {y_.shape}')


# ----------------------------
# Training Function
# ----------------------------
def run_csvae(
    beta1: float = 20.0,
    beta2: float = 1.0,
    beta3: float = 0.2,
    beta4: float = 1000.0,
    epochs: int = 20,
    device: torch.device = torch.device('cpu')
) -> tuple:
    # Model dimensions
    encoder_w = EncoderW(dim_in=4, dim_out=2).to(device)
    encoder_z = EncoderZ(dim_in=3, dim_out=2).to(device)
    decoder_x = DecoderX(dim_in=4, dim_out=3).to(device)
    decoder_y = DecoderY(dim_in=2, n_classes=2).to(device)

    optimizer1 = torch.optim.Adam(
        chain(decoder_x.parameters(), encoder_w.parameters(), encoder_z.parameters()),
        lr=1e-3
    )
    optimizer2 = torch.optim.Adam(decoder_y.parameters(), lr=1e-3)

    dataset = list(zip(torch.from_numpy(X_), torch.from_numpy(y_)))
    trainloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=256)

    prior_logvar_w = np.log(0.01)

    with trange(epochs) as t:
        for epoch in t:
            t.set_description(f'Epoch {epoch}')
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_onehot = y_batch.unsqueeze(-1).float()

                # Forward pass
                mu_z, logvar_z, z = encoder_z(x_batch)
                mu_w, logvar_w, w = encoder_w(x_batch, y_onehot)

                mu_x, logvar_x, _ = decoder_x(w, z)
                y_prob = decoder_y(z)

                # KL divergences
                kl1 = _kl_gaussian(
                    mu_w, logvar_w,
                    torch.zeros_like(mu_w),
                    torch.full_like(logvar_w, prior_logvar_w)
                )
                kl0 = _kl_gaussian(
                    mu_w, logvar_w,
                    torch.full_like(mu_w, 3.0),
                    torch.full_like(logvar_w, prior_logvar_w)
                )

                # Loss components
                recon_loss = torch.sum((x_batch - mu_x) ** 2, dim=-1)
                kl_w = torch.where(y_batch == 1, kl1, kl0)
                kl_z = _kl_gaussian(
                    mu_z, logvar_z,
                    torch.zeros_like(mu_z),
                    torch.zeros_like(logvar_z)
                )
                # Clamp to avoid log(0)
                y_prob = torch.clamp(y_prob, 1e-8, 1.0)
                entropy_y = torch.sum(y_prob * torch.log(y_prob), dim=-1)

                loss1 = (
                    beta1 * recon_loss +
                    beta2 * kl_w +
                    beta3 * kl_z +
                    beta4 * entropy_y
                ).sum()

                # Classification loss (cross-entropy via manual log)
                loss2 = 100.0 * torch.where(
                    y_batch == 1,
                    -torch.log(y_prob[:, 1]),
                    -torch.log(y_prob[:, 0])
                ).sum()

                total_loss = loss1 + loss2

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                total_loss.backward()
                optimizer1.step()
                optimizer2.step()

                t.set_postfix(
                    loss=total_loss.item(),
                    y_max=y_prob.max().item(),
                    y_min=y_prob.min().item()
                )

    return encoder_w, encoder_z, decoder_x, decoder_y


# ----------------------------
# Visualization Functions
# ----------------------------
def plot_reconstruction(pred_x: np.ndarray, y_: np.ndarray, X_original: np.ndarray):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(X_original[y_ == 0, 0], X_original[y_ == 0, 1], X_original[y_ == 0, 2],
               s=1, c='forestgreen', alpha=0.7, label='Original class 0')
    ax.scatter(X_original[y_ == 1, 0], X_original[y_ == 1, 1], X_original[y_ == 1, 2],
               s=1, c='purple', alpha=0.7, label='Original class 1')
    ax.view_init(0, 30)
    ax.set_title('Original Data')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(pred_x[y_ == 0, 0], pred_x[y_ == 0, 1], pred_x[y_ == 0, 2],
               s=1, c='forestgreen', alpha=0.7, label='Reconstructed class 0')
    ax.scatter(pred_x[y_ == 1, 0], pred_x[y_ == 1, 1], pred_x[y_ == 1, 2],
               s=1, c='purple', alpha=0.7, label='Reconstructed class 1')
    ax.view_init(0, 30)
    ax.set_title('Reconstructed Data')
    ax.legend()
    plt.tight_layout()


def plot_latent_subspaces(z_: np.ndarray, w_: np.ndarray, y_: np.ndarray, figsize=(15, 8)):
    fig = plt.figure(figsize=figsize)
    titles = [r'$(Z_1, Z_2)$', r'$(W_1, W_2)$', r'$(W_1, Z_2)$', r'$(Z_1, W_2)$']
    coords = [
        (z_[:, 0], z_[:, 1]),
        (w_[:, 0], w_[:, 1]),
        (w_[:, 0], z_[:, 1]),
        (z_[:, 0], w_[:, 1]),
    ]
    for i, (x, y) in enumerate(coords, 1):
        plt.subplot(2, 2, i)
        plt.scatter(x[y_ == 0], y[y_ == 0], s=1, c='forestgreen', alpha=0.3)
        plt.scatter(x[y_ == 1], y[y_ == 1], s=1, c='purple', alpha=0.3)
        plt.title(titles[i - 1], fontsize=15)
    plt.tight_layout()


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.from_numpy(X_).to(device)
    y_tensor = torch.from_numpy(y_).to(device)

    # ✅ Successful run
    models_good = run_csvae(beta1=20., beta2=1., beta3=0.2, beta4=1000, epochs=20, device=device)
    with torch.no_grad():
        z_good, _, _ = models_good[1](X_tensor)
        w_good, _, _ = models_good[0](X_tensor, y_tensor.unsqueeze(-1).float())
        pred_x_good, _, _ = models_good[2](w_good, z_good)
    plot_reconstruction(pred_x_good.cpu().numpy(), y_, X_)
    plt.suptitle('Successful Reconstruction\n[Correct Hyperparameters]')
    plt.show()

    plot_latent_subspaces(z_good.cpu().numpy(), w_good.cpu().numpy(), y_)
    plt.suptitle('Latent Spaces – Successful Run')
    plt.show()

    # ❌ Failure run (beta3 too large)
    models_bad = run_csvae(beta1=20., beta2=1., beta3=2000., beta4=1000, epochs=20, device=device)
    with torch.no_grad():
        z_bad, _, _ = models_bad[1](X_tensor)
        w_bad, _, _ = models_bad[0](X_tensor, y_tensor.unsqueeze(-1).float())
        pred_x_bad, _, _ = models_bad[2](w_bad, z_bad)
    plot_reconstruction(pred_x_bad.cpu().numpy(), y_, X_)
    plt.suptitle('Unsuccessful Reconstruction\n[Excessive beta3]')
    plt.show()

    plot_latent_subspaces(z_bad.cpu().numpy(), w_bad.cpu().numpy(), y_)
    plt.suptitle('Latent Spaces – Failed Run')
    plt.show()