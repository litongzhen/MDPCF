# Adapted from TABSYN; modified with CSVAE branches and objectives for MDPCE.
# See THIRD_PARTY_NOTICES.md and licenses/TABSYN-APACHE-2.0.txt.
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import math


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, coefficient):
        ctx.coefficient = coefficient
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(input_tensor, coefficient=1.0):
    return _GradientReversal.apply(input_tensor, float(coefficient))


class Tokenizer(nn.Module):

    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization="kaiming"):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ["xavier", "kaiming"]

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == "xavier" and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):

        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        initialization="kaiming",
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    "linear0": nn.Linear(d_token, d_hidden),
                    "linear1": nn.Linear(d_hidden, d_token),
                    "norm1": make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x):
        for layer in self.layers:
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer["linear0"](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer["linear1"](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class MLPTokenBackbone(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_token,
        num_layers=2,
        factor=32,
        parameter_budget=None,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        input_dim = n_tokens * d_token
        depth = max(1, int(num_layers))
        if parameter_budget is None:
            hidden_dim = max(input_dim, int(input_dim * factor))
        else:
            candidates = range(1, max(2, int(parameter_budget) + 1))
            hidden_dim = min(
                candidates,
                key=lambda width: abs(
                    (depth - 1) * width * width
                    + 2 * input_dim * width
                    + depth * width
                    + input_dim
                    - parameter_budget
                ),
            )
        self.hidden_dim = hidden_dim

        layers = []
        for layer_idx in range(depth):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        h = x.reshape(batch_size, self.n_tokens * self.d_token)
        h = self.net(h)
        return h.reshape(batch_size, self.n_tokens, self.d_token)


class Reconstructor(nn.Module):
    def __init__(
        self,
        d_numerical,
        categories,
        d_token,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories or []
        self.d_token = d_token
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau
        self.sigmoid = sigmoid

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn_init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList([nn.Linear(d_token, d) for d in self.categories])
        for recon in self.cat_recons:
            nn_init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))

    def forward(self, h, deterministic=False):
        h_num = h[:, : self.d_numerical]
        h_cat = h[:, self.d_numerical :]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        if self.sigmoid:
            recon_x_num = torch.sigmoid(recon_x_num)

        recon_x_cat = []
        for idx, recon in enumerate(self.cat_recons):
            logits = recon(h_cat[:, idx])
            if self.gumbel_softmax and not deterministic:
                recon_x_cat.append(F.gumbel_softmax(logits, tau=self.tau, hard=True))
            elif self.gumbel_softmax:
                recon_x_cat.append(torch.softmax(logits / self.tau, dim=-1))
            else:
                recon_x_cat.append(logits)

        return recon_x_num, recon_x_cat


class CSVAEEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        dim_w=16,
        n_head=1,
        factor=4,
        bias=True,
        backbone="transformer",
        w_structure="global",
        mlp_parameter_match=False,
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown CSVAE backbone: {backbone}")
        self.categories = categories or []
        self.n_feature_tokens = d_numerical + len(self.categories)
        self.w_structure = w_structure
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.backbone_name = backbone
        if backbone == "transformer":
            self.backbone = Transformer(num_layers, d_token, n_head, factor)
        else:
            parameter_budget = None
            if mlp_parameter_match:
                reference = Transformer(num_layers, d_token, n_head, factor)
                parameter_budget = sum(
                    parameter.numel() for parameter in reference.parameters()
                )
            self.backbone = MLPTokenBackbone(
                self.Tokenizer.n_tokens,
                d_token,
                num_layers=num_layers,
                factor=factor,
                parameter_budget=parameter_budget,
            )
        self.z_mu = nn.Linear(d_token, dim_z)
        self.z_logvar = nn.Linear(d_token, dim_z)
        self.w_mu = nn.Linear(d_token + 1, dim_w)
        self.w_logvar = nn.Linear(d_token + 1, dim_w)
        adversary_input_dim = self.n_feature_tokens * dim_z
        adversary_hidden_dim = max(16, adversary_input_dim // 2)
        self.z_adversary = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(adversary_input_dim, adversary_hidden_dim),
            nn.ReLU(),
            nn.Linear(adversary_hidden_dim, 2),
        )

    def encode(self, x_num, x_cat):
        tokens = self.Tokenizer(x_num, x_cat)
        hidden = self.backbone(tokens)
        return hidden[:, 1:, :]

    def forward(
        self,
        x_num,
        x_cat,
        y=None,
        deterministic=False,
        adversary_weight=1.0,
    ):
        feature_tokens = self.encode(x_num, x_cat)
        mu_z = self.z_mu(feature_tokens)
        logvar_z = self.z_logvar(feature_tokens)
        z = mu_z if deterministic else mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)
        z_label_logits = self.z_adversary(gradient_reverse(z, adversary_weight))

        if y is None:
            y_feat = torch.zeros(
                feature_tokens.shape[0],
                1,
                dtype=feature_tokens.dtype,
                device=feature_tokens.device,
            )
        else:
            if y.dim() == 1:
                y_feat = y.float().unsqueeze(-1)
            else:
                y_feat = y.float()
        if self.w_structure == "global":
            pooled_features = feature_tokens.mean(dim=1)
            w_input = torch.cat([pooled_features, y_feat], dim=-1)
        else:
            token_labels = y_feat.unsqueeze(1).expand(
                -1, self.n_feature_tokens, -1
            )
            w_input = torch.cat([feature_tokens, token_labels], dim=-1)
        mu_w = self.w_mu(w_input)
        logvar_w = self.w_logvar(w_input)
        w = mu_w if deterministic else mu_w + torch.randn_like(mu_w) * torch.exp(0.5 * logvar_w)
        return mu_z, logvar_z, z, mu_w, logvar_w, w, z_label_logits


class CSVAE_Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        n_head,
        factor,
        dim_z=16,
        dim_w=16,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
        mlp_parameter_match=False,
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown CSVAE backbone: {backbone}")
        self.d_numerical = d_numerical
        self.categories = categories or []
        self.n_feature_tokens = d_numerical + len(self.categories)
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.fuse = nn.Linear(dim_z + dim_w, d_token)
        self.backbone_name = backbone
        if backbone == "transformer":
            self.decoder = Transformer(num_layers, d_token, n_head, factor)
        else:
            parameter_budget = None
            if mlp_parameter_match:
                reference = Transformer(num_layers, d_token, n_head, factor)
                parameter_budget = sum(
                    parameter.numel() for parameter in reference.parameters()
                )
            self.decoder = MLPTokenBackbone(
                self.n_feature_tokens,
                d_token,
                num_layers=num_layers,
                factor=factor,
                parameter_budget=parameter_budget,
            )
        self.reconstructor = Reconstructor(
            d_numerical,
            categories,
            d_token,
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
        )

    def forward(self, z, w, deterministic=False):
        if w.dim() == 2:
            w = w.unsqueeze(1).expand(-1, z.shape[1], -1)
        h = self.fuse(torch.cat([z, w], dim=-1))
        h = self.decoder(h)
        return self.reconstructor(h, deterministic=deterministic)


class Model_CSVAE(nn.Module):
    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        dim_w=16,
        n_head=1,
        factor=4,
        bias=True,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
        w_structure="global",
        mlp_parameter_match=False,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.dim_w = dim_w
        prior_mu = torch.stack(
            [torch.full((dim_w,), -1.0), torch.full((dim_w,), 1.0)]
        )
        self.w_prior_mu = nn.Parameter(prior_mu)
        self.w_prior_logvar = nn.Parameter(torch.zeros(2, dim_w))
        self.encoder = CSVAEEncoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=dim_z,
            dim_w=dim_w,
            n_head=n_head,
            factor=factor,
            bias=bias,
            backbone=backbone,
            w_structure=w_structure,
            mlp_parameter_match=mlp_parameter_match,
        )
        self.decoder = CSVAE_Decoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head,
            factor,
            dim_z=dim_z,
            dim_w=dim_w,
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
            backbone=backbone,
            mlp_parameter_match=mlp_parameter_match,
        )
        self.z_adversary = self.encoder.z_adversary

    def class_prior(self, y, n_tokens=None):
        label_index = y.long().reshape(-1)
        prior_mu = self.w_prior_mu[label_index]
        prior_logvar = self.w_prior_logvar[label_index]
        if n_tokens is not None:
            prior_mu = prior_mu.unsqueeze(1).expand(-1, n_tokens, -1)
            prior_logvar = prior_logvar.unsqueeze(1).expand(-1, n_tokens, -1)
        return prior_mu, prior_logvar

    def forward(
        self,
        x_num,
        x_cat,
        y=None,
        deterministic=False,
        adversary_weight=1.0,
    ):
        mu_z, logvar_z, z, mu_w, logvar_w, w, z_label_logits = self.encoder(
            x_num,
            x_cat,
            y,
            deterministic=deterministic,
            adversary_weight=adversary_weight,
        )
        recon_x_num, recon_x_cat = self.decoder(z, w, deterministic=deterministic)
        return (
            recon_x_num,
            recon_x_cat,
            mu_z,
            logvar_z,
            z,
            mu_w,
            logvar_w,
            w,
            z_label_logits,
        )


class StandardVAEEncoder(nn.Module):
    """Token-based encoder with a single, label-free latent branch."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        bias=True,
        backbone="transformer",
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown VAE backbone: {backbone}")
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        if backbone == "transformer":
            self.backbone = Transformer(num_layers, d_token, n_head, factor)
        else:
            self.backbone = MLPTokenBackbone(
                self.Tokenizer.n_tokens, d_token, num_layers=num_layers, factor=factor
            )
        self.z_mu = nn.Linear(d_token, dim_z)
        self.z_logvar = nn.Linear(d_token, dim_z)

    def forward(self, x_num, x_cat, deterministic=False):
        hidden = self.backbone(self.Tokenizer(x_num, x_cat))[:, 1:, :]
        mu_z = self.z_mu(hidden)
        logvar_z = self.z_logvar(hidden)
        z = mu_z if deterministic else mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)
        return mu_z, logvar_z, z


class StandardVAEDecoder(nn.Module):
    """Token decoder that reconstructs tabular inputs from z alone."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown VAE backbone: {backbone}")
        self.n_feature_tokens = d_numerical + len(categories or [])
        self.dim_z = dim_z
        self.project = nn.Linear(dim_z, d_token)
        if backbone == "transformer":
            self.decoder = Transformer(num_layers, d_token, n_head, factor)
        else:
            self.decoder = MLPTokenBackbone(
                self.n_feature_tokens, d_token, num_layers=num_layers, factor=factor
            )
        self.reconstructor = Reconstructor(
            d_numerical,
            categories,
            d_token,
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
        )

    def forward(self, z, deterministic=False):
        if z.dim() == 2:
            z = z.reshape(z.size(0), self.n_feature_tokens, self.dim_z)
        return self.reconstructor(
            self.decoder(self.project(z)), deterministic=deterministic
        )


class Model_StandardVAE(nn.Module):
    """A genuine standard VAE without residual or label-conditioned branches."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        bias=True,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
    ):
        super().__init__()
        self.encoder = StandardVAEEncoder(
            num_layers, d_numerical, categories, d_token, dim_z, n_head, factor, bias, backbone
        )
        self.decoder = StandardVAEDecoder(
            num_layers, d_numerical, categories, d_token, dim_z, n_head, factor,
            gumbel_softmax, tau, sigmoid, backbone
        )

    def forward(self, x_num, x_cat, deterministic=False):
        mu_z, logvar_z, z = self.encoder(x_num, x_cat, deterministic=deterministic)
        recon_x_num, recon_x_cat = self.decoder(z, deterministic=deterministic)
        return recon_x_num, recon_x_cat, mu_z, logvar_z, z


class StandardVAEEncoderModel(nn.Module):
    """Sampling-facing wrapper for the standard VAE encoder."""

    def __init__(
        self, num_layers, d_numerical, categories, d_token, n_head, factor,
        bias=True, backbone="transformer",
    ):
        super().__init__()
        self.model = StandardVAEEncoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=d_token,
            n_head=n_head,
            factor=factor,
            bias=bias,
            backbone=backbone,
        )

    def load_weights(self, pretrained_vae):
        self.model.load_state_dict(pretrained_vae.encoder.state_dict())

    def forward(self, x_num, x_cat):
        _, _, z = self.model(x_num, x_cat)
        return z


class StandardVAEDecoderModel(nn.Module):
    """Sampling-facing decoder that intentionally ignores the legacy w input."""

    def __init__(
        self, num_layers, d_numerical, categories, d_token, n_head, factor,
        gumbel_softmax=False, sigmoid=False, tau=1.0, backbone="transformer",
    ):
        super().__init__()
        self.n_feature_tokens = d_numerical + len(categories or [])
        self.dim_z = d_token
        self.model = StandardVAEDecoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=d_token,
            n_head=n_head,
            factor=factor,
            gumbel_softmax=gumbel_softmax,
            sigmoid=sigmoid,
            tau=tau,
            backbone=backbone,
        )

    def load_weights(self, pretrained_vae):
        self.model.load_state_dict(pretrained_vae.decoder.state_dict())

    def forward(self, z, w=None):
        if z.dim() == 2:
            z = z.reshape(z.size(0), self.n_feature_tokens, self.dim_z)
        return self.model(z)


def _expand_binary_condition(condition, batch_size, device):
    condition = torch.as_tensor(condition, device=device).long().reshape(-1)
    if condition.numel() == 1:
        condition = condition.expand(batch_size)
    if condition.numel() != batch_size:
        raise ValueError(
            f"Expected one condition or {batch_size} conditions; got {condition.numel()}."
        )
    if torch.any((condition < 0) | (condition > 1)):
        raise ValueError("CVAE condition must be binary (0 or 1).")
    return condition


class ConditionalVAEEncoder(nn.Module):
    """Token encoder for q(z | x, y) with a binary class condition."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        bias=True,
        backbone="transformer",
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown CVAE backbone: {backbone}")
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.condition_embedding = nn.Embedding(2, d_token)
        if backbone == "transformer":
            self.backbone = Transformer(num_layers, d_token, n_head, factor)
        else:
            self.backbone = MLPTokenBackbone(
                self.Tokenizer.n_tokens, d_token, num_layers=num_layers, factor=factor
            )
        self.z_mu = nn.Linear(d_token, dim_z)
        self.z_logvar = nn.Linear(d_token, dim_z)

    def forward(self, x_num, x_cat, condition, deterministic=False):
        tokens = self.Tokenizer(x_num, x_cat)
        labels = _expand_binary_condition(condition, tokens.shape[0], tokens.device)
        tokens = tokens + self.condition_embedding(labels).unsqueeze(1)
        hidden = self.backbone(tokens)[:, 1:, :]
        mu_z = self.z_mu(hidden)
        logvar_z = self.z_logvar(hidden)
        z = mu_z if deterministic else mu_z + torch.randn_like(mu_z) * torch.exp(
            0.5 * logvar_z
        )
        return mu_z, logvar_z, z


class ConditionalVAEDecoder(nn.Module):
    """Token decoder for p(x | z, y) with a binary class condition."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
    ):
        super().__init__()
        if backbone not in {"transformer", "mlp"}:
            raise ValueError(f"Unknown CVAE backbone: {backbone}")
        self.n_feature_tokens = d_numerical + len(categories or [])
        self.dim_z = dim_z
        self.project = nn.Linear(dim_z, d_token)
        self.condition_embedding = nn.Embedding(2, d_token)
        if backbone == "transformer":
            self.decoder = Transformer(num_layers, d_token, n_head, factor)
        else:
            self.decoder = MLPTokenBackbone(
                self.n_feature_tokens, d_token, num_layers=num_layers, factor=factor
            )
        self.reconstructor = Reconstructor(
            d_numerical,
            categories,
            d_token,
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
        )

    def forward(self, z, condition, deterministic=False):
        if z.dim() == 2:
            z = z.reshape(z.size(0), self.n_feature_tokens, self.dim_z)
        labels = _expand_binary_condition(condition, z.shape[0], z.device)
        hidden = self.project(z) + self.condition_embedding(labels).unsqueeze(1)
        return self.reconstructor(
            self.decoder(hidden), deterministic=deterministic
        )


class Model_ConditionalVAE(nn.Module):
    """Single-latent CVAE implementing q(z | x, y) and p(x | z, y)."""

    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        dim_z=16,
        n_head=1,
        factor=4,
        bias=True,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
        backbone="transformer",
    ):
        super().__init__()
        self.encoder = ConditionalVAEEncoder(
            num_layers, d_numerical, categories, d_token, dim_z,
            n_head, factor, bias, backbone
        )
        self.decoder = ConditionalVAEDecoder(
            num_layers, d_numerical, categories, d_token, dim_z,
            n_head, factor, gumbel_softmax, tau, sigmoid, backbone
        )

    def forward(self, x_num, x_cat, condition, deterministic=False):
        mu_z, logvar_z, z = self.encoder(
            x_num, x_cat, condition, deterministic=deterministic
        )
        recon_x_num, recon_x_cat = self.decoder(
            z, condition, deterministic=deterministic
        )
        return recon_x_num, recon_x_cat, mu_z, logvar_z, z


class ConditionalVAEEncoderModel(nn.Module):
    """Sampling-facing wrapper for the conditional VAE encoder."""

    def __init__(
        self, num_layers, d_numerical, categories, d_token, n_head, factor,
        bias=True, backbone="transformer",
    ):
        super().__init__()
        self.model = ConditionalVAEEncoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=d_token,
            n_head=n_head,
            factor=factor,
            bias=bias,
            backbone=backbone,
        )

    def load_weights(self, pretrained_vae):
        self.model.load_state_dict(pretrained_vae.encoder.state_dict())

    def forward(self, x_num, x_cat, condition):
        _, _, z = self.model(x_num, x_cat, condition, deterministic=True)
        return z


class ConditionalVAEDecoderModel(nn.Module):
    """Sampling-facing wrapper whose second input is the fixed class condition."""

    def __init__(
        self, num_layers, d_numerical, categories, d_token, n_head, factor,
        gumbel_softmax=False, sigmoid=False, tau=1.0, backbone="transformer",
    ):
        super().__init__()
        self.n_feature_tokens = d_numerical + len(categories or [])
        self.dim_z = d_token
        self.model = ConditionalVAEDecoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=d_token,
            n_head=n_head,
            factor=factor,
            gumbel_softmax=gumbel_softmax,
            sigmoid=sigmoid,
            tau=tau,
            backbone=backbone,
        )

    def load_weights(self, pretrained_vae):
        self.model.load_state_dict(pretrained_vae.decoder.state_dict())

    def forward(self, z, condition):
        if z.dim() == 2:
            z = z.reshape(z.size(0), self.n_feature_tokens, self.dim_z)
        return self.model(z, condition)


class Encoder_model(nn.Module):
    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        n_head,
        factor,
        bias=True,
        backbone="transformer",
        dim_w=2,
        mlp_parameter_match=False,
    ):
        super(Encoder_model, self).__init__()
        self.model = CSVAEEncoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            dim_z=d_token,
            dim_w=dim_w,
            n_head=n_head,
            factor=factor,
            bias=bias,
            backbone=backbone,
            mlp_parameter_match=mlp_parameter_match,
        )

    def load_weights(self, pretrained_csvae):
        self.model.load_state_dict(pretrained_csvae.encoder.state_dict())

    def forward(self, x_num, x_cat):
        _, _, z, _, _, _, _ = self.model(x_num, x_cat)
        return z


class Decoder_model(nn.Module):
    def __init__(
        self,
        num_layers,
        d_numerical,
        categories,
        d_token,
        n_head,
        factor,
        bias=True,
        gumbel_softmax=False,
        sigmoid=False,
        tau=1.0,
        backbone="transformer",
        dim_w=2,
        mlp_parameter_match=False,
    ):
        super(Decoder_model, self).__init__()
        self.n_feature_tokens = d_numerical + len(categories or [])
        self.dim_z = d_token
        self.dim_w = dim_w
        self.model = CSVAE_Decoder(
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head,
            factor,
            dim_z=d_token,
            dim_w=dim_w,
            gumbel_softmax=gumbel_softmax,
            sigmoid=sigmoid,
            tau=tau,
            backbone=backbone,
            mlp_parameter_match=mlp_parameter_match,
        )

    def load_weights(self, pretrained_csvae):
        self.model.load_state_dict(pretrained_csvae.decoder.state_dict())

    def forward(self, z, w=None):
        if w is None:
            z_flat = z.reshape(z.size(0), -1)
            half = z_flat.size(-1) // 2
            z = z_flat[:, :half].reshape(z.size(0), self.n_feature_tokens, self.dim_z)
            w = z_flat[:, half:].reshape(z.size(0), self.n_feature_tokens, self.dim_w)
        else:
            if z.dim() == 2:
                z = z.reshape(z.size(0), self.n_feature_tokens, self.dim_z)
            if w.dim() == 2 and w.shape[-1] != self.dim_w:
                w = w.reshape(w.size(0), self.n_feature_tokens, self.dim_w)
        x_hat_num, x_hat_cat = self.model(z, w)
        return x_hat_num, x_hat_cat


class BBMLPCLF(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_size=1000):
        super(BBMLPCLF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)  # Apply sigmoid to the output
        return x
