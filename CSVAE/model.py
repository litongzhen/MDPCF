from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_gaussian(mu1: torch.Tensor, logvar1: torch.Tensor,
                 mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence KL(N(mu1, var1) || N(mu2, var2)) per sample.
    Returns: shape (batch_size,)
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0)
    return torch.sum(kl, dim=-1)


class EncoderW(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_units: int = 32):
        super().__init__()
        self.dim_out = dim_out
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(dim_in, n_units)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(n_units, n_units)),
            ('relu2', nn.ReLU()),
            ('final_linear', nn.Linear(n_units, 2 * dim_out))
        ]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(torch.cat((x, y), dim=-1))
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        w = mu + eps * std
        return mu, logvar, w


class EncoderZ(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_units: int = 32):
        super().__init__()
        self.dim_out = dim_out
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(dim_in, n_units)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(n_units, n_units)),
            ('relu2', nn.ReLU()),
            ('final_linear', nn.Linear(n_units, 2 * dim_out))
        ]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z


class DecoderX(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_units: int = 32):
        super().__init__()
        self.dim_out = dim_out
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(dim_in, n_units)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(n_units, n_units)),
            ('relu2', nn.ReLU()),
            ('final_linear', nn.Linear(n_units, 2 * dim_out))
        ]))

    def forward(self, w: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(torch.cat((w, z), dim=-1))
        mu = out[:, :self.dim_out]
        logvar = out[:, self.dim_out:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        return mu, logvar, x


class DecoderY(nn.Module):
    def __init__(self, dim_in: int, n_classes: int, n_units: int = 32):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(dim_in, n_units)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(n_units, n_units)),
            ('relu2', nn.ReLU()),
            ('final_linear', nn.Linear(n_units, n_classes))
        ]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.model(z)
        # Use log_softmax for numerical stability if needed, but softmax is fine here
        return F.softmax(logits, dim=-1)