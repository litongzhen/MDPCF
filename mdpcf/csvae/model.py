import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import math


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
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        activation="relu",
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
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

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
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                # for the last attention, it is enough to process only [CLS]
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
    ):
        super().__init__()
        hidden_dim = d_token * factor
        self.categories = categories or []
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.backbone = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.project = nn.Sequential(
            nn.Linear((d_numerical + 1 + len(self.categories)) * d_token, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu = nn.Linear(hidden_dim, dim_z)
        self.z_logvar = nn.Linear(hidden_dim, dim_z)
        self.w_mu = nn.Linear(hidden_dim + 1, dim_w)
        self.w_logvar = nn.Linear(hidden_dim + 1, dim_w)
        self.y_head = nn.Linear(dim_z, 2)

    def encode(self, x_num, x_cat):
        tokens = self.Tokenizer(x_num, x_cat)
        hidden = self.backbone(tokens)
        hidden = hidden.reshape(hidden.size(0), -1)
        hidden = self.project(hidden)
        return hidden

    def forward(self, x_num, x_cat, y=None):
        hidden = self.encode(x_num, x_cat)
        mu_z = self.z_mu(hidden)
        logvar_z = self.z_logvar(hidden)
        z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)
        y_logits = self.y_head(z)

        if y is None:
            y_feat = torch.softmax(y_logits.detach(), dim=-1)[:, 1:2]
        else:
            if y.dim() == 1:
                y_feat = y.float().unsqueeze(-1)
            else:
                y_feat = y.float()
        w_input = torch.cat([hidden, y_feat], dim=-1)
        mu_w = self.w_mu(w_input)
        logvar_w = self.w_logvar(w_input)
        w = mu_w + torch.randn_like(mu_w) * torch.exp(0.5 * logvar_w)
        return mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits


class CSVAE_Decoder(nn.Module):
    def __init__(
        self,
        d_numerical,
        categories,
        dim_z=16,
        dim_w=16,
        hidden_dim=64,
        gumbel_softmax=False,
        tau=1.0,
        sigmoid=False,
    ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories or []
        self.gumbel_softmax = gumbel_softmax
        self.tau = tau
        self.sigmoid = sigmoid
        self.shared = nn.Sequential(
            nn.Linear(dim_z + dim_w, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.num_head = nn.Linear(hidden_dim, d_numerical)
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, d) for d in self.categories])

    def forward(self, z, w):
        h = self.shared(torch.cat([z, w], dim=-1))
        recon_x_num = self.num_head(h)
        if self.sigmoid:
            recon_x_num = torch.sigmoid(recon_x_num)

        recon_x_cat = []
        for head in self.cat_heads:
            logits = head(h)
            if self.gumbel_softmax:
                recon_x_cat.append(F.gumbel_softmax(logits, tau=self.tau, hard=True))
            else:
                recon_x_cat.append(logits)
        return recon_x_num, recon_x_cat


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
    ):
        super().__init__()
        self.dim_z = dim_z
        self.dim_w = dim_w
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
        )
        self.decoder = CSVAE_Decoder(
            d_numerical,
            categories,
            dim_z=dim_z,
            dim_w=dim_w,
            hidden_dim=max(64, d_token * factor),
            gumbel_softmax=gumbel_softmax,
            tau=tau,
            sigmoid=sigmoid,
        )
        self.y_head = self.encoder.y_head

    def forward(self, x_num, x_cat, y=None):
        mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits = self.encoder(x_num, x_cat, y)
        recon_x_num, recon_x_cat = self.decoder(z, w)
        return recon_x_num, recon_x_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits


class Encoder_model(nn.Module):
    def __init__(
        self, num_layers, d_numerical, categories, d_token, n_head, factor, bias=True
    ):
        super(Encoder_model, self).__init__()
        self.model = CSVAEEncoder(
            num_layers, d_numerical, categories, d_token, dim_z=d_token, dim_w=d_token, n_head=n_head, factor=factor, bias=bias
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
    ):
        super(Decoder_model, self).__init__()
        self.model = CSVAE_Decoder(
            d_numerical,
            categories,
            dim_z=d_token,
            dim_w=d_token,
            hidden_dim=max(64, d_token * factor),
            gumbel_softmax=gumbel_softmax,
            sigmoid=sigmoid,
            tau=tau,
        )

    def load_weights(self, pretrained_csvae):
        self.model.load_state_dict(pretrained_csvae.decoder.state_dict())

    def forward(self, z, w=None):
        if z.dim() == 3:
            z = z[:, 1:, :].reshape(z.size(0), -1)
        if w is None:
            half = z.size(-1) // 2
            z, w = z[:, :half], z[:, half:]
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
