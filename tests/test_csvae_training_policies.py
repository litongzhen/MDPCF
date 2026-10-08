import numpy as np
import pytest
import torch
from pathlib import Path

from mdpcf.csvae.main import (
    adversary_confusion_loss,
    adversary_class_weights,
    z_adversary_logits,
    beta_for_epoch,
    conditional_gaussian_kl,
    split_training_validation_indices,
)
from mdpcf.csvae.model import MLPTokenBackbone, Model_CSVAE, Transformer


def test_csvae_training_initializes_pre_decoder_in_csvae_branch():
    source = (
        Path(__file__).resolve().parents[1] / "mdpcf" / "csvae" / "main.py"
    ).read_text(encoding="utf-8")
    csvae_branch = source.rsplit('elif latent_model == "csvae":', 1)[1].split(
        "else:", 1
    )[0]

    assert "pre_decoder = Decoder_model(" in csvae_branch


def test_fixed_beta_returns_target_for_every_epoch():
    assert beta_for_epoch("fixed", 0, 1e-3, 300, 1e-2) == pytest.approx(1e-3)
    assert beta_for_epoch("fixed", 999, 1e-3, 300, 1e-2) == pytest.approx(1e-3)


def test_linear_warmup_reaches_target_and_stays_there():
    assert beta_for_epoch("linear_warmup", 0, 1e-3, 300, 1e-2) == 0.0
    assert beta_for_epoch("linear_warmup", 150, 1e-3, 300, 1e-2) == pytest.approx(
        1e-3 * 150 / 299
    )
    assert beta_for_epoch("linear_warmup", 299, 1e-3, 300, 1e-2) == pytest.approx(1e-3)
    assert beta_for_epoch("linear_warmup", 999, 1e-3, 300, 1e-2) == pytest.approx(1e-3)


def test_adaptive_beta_uses_current_adaptive_value():
    assert beta_for_epoch("adaptive", 100, 1e-3, 300, 7e-4) == pytest.approx(7e-4)


def test_adversary_class_weights_balance_binary_label_mass():
    labels = torch.tensor([0, 0, 0, 1])

    weights = adversary_class_weights(labels, num_classes=2)

    assert weights.tolist() == pytest.approx([2.0 / 3.0, 2.0])
    assert 3 * weights[0] == pytest.approx(weights[1])


def test_adversary_class_weights_reject_missing_classes():
    with pytest.raises(ValueError, match="all adversary classes"):
        adversary_class_weights(torch.zeros(4, dtype=torch.long), num_classes=2)


def test_adversary_confusion_loss_prefers_uniform_predictions():
    uniform_logits = torch.zeros(4, 2)
    confident_logits = torch.tensor([[8.0, -8.0]]).repeat(4, 1)

    uniform_loss = adversary_confusion_loss(uniform_logits)
    confident_loss = adversary_confusion_loss(confident_logits)

    assert uniform_loss.item() == pytest.approx(-np.log(2), abs=1e-6)
    assert uniform_loss < confident_loss


def test_normalized_z_adversary_is_invariant_to_shared_affine_scale():
    model = Model_CSVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        dim_w=4,
        n_head=1,
        factor=2,
    ).eval()
    z = torch.randn(3, 2, 4)

    original = z_adversary_logits(model, z, normalize=True)
    rescaled = z_adversary_logits(model, z * 3.0 + 5.0, normalize=True)

    assert torch.allclose(original, rescaled, atol=1e-5)


def test_matched_mlp_backbone_uses_transformer_parameter_budget():
    transformer = Transformer(n_layers=2, d_token=4, n_heads=1, d_ffn_factor=32)
    budget = sum(parameter.numel() for parameter in transformer.parameters())

    mlp = MLPTokenBackbone(
        n_tokens=13,
        d_token=4,
        num_layers=2,
        factor=32,
        parameter_budget=budget,
    )
    mlp_parameters = sum(parameter.numel() for parameter in mlp.parameters())

    assert mlp.hidden_dim < 13 * 4
    assert abs(mlp_parameters - budget) / budget < 0.1


def test_training_validation_split_is_seeded_disjoint_and_exhaustive():
    train_a, val_a = split_training_validation_indices(100, 0.1, 42)
    train_b, val_b = split_training_validation_indices(100, 0.1, 42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert len(train_a) == 90
    assert len(val_a) == 10
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(range(100))


def test_csvAE_deterministic_forward_repeats_latents_and_categorical_outputs():
    model = Model_CSVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        dim_w=4,
        n_head=1,
        factor=2,
        gumbel_softmax=True,
    ).eval()
    x_num = torch.tensor([[0.2], [0.8]])
    x_cat = torch.tensor([[0], [1]])
    y = torch.tensor([0, 1])

    first = model(x_num, x_cat, y=y, deterministic=True)
    second = model(x_num, x_cat, y=y, deterministic=True)

    assert torch.equal(first[4], second[4])
    assert torch.equal(first[7], second[7])
    assert torch.equal(first[1][0], second[1][0])
    assert torch.allclose(first[1][0].sum(dim=-1), torch.ones(2))
    assert torch.all(first[1][0] >= 0)


def test_csvae_z_is_label_free_and_w_is_label_conditioned():
    model = Model_CSVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        dim_w=4,
        n_head=1,
        factor=2,
    ).eval()
    x_num = torch.tensor([[0.2], [0.8]])
    x_cat = torch.tensor([[0], [1]])

    encoded_zero = model.encoder(
        x_num, x_cat, y=torch.zeros(2), deterministic=True
    )
    encoded_one = model.encoder(
        x_num, x_cat, y=torch.ones(2), deterministic=True
    )

    assert torch.allclose(encoded_zero[0], encoded_one[0])
    assert not torch.allclose(encoded_zero[3], encoded_one[3])
    assert encoded_zero[3].shape == (2, 4)


def test_csvae_class_prior_selects_label_specific_w_gaussians():
    model = Model_CSVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        dim_w=4,
        n_head=1,
        factor=2,
    )

    prior_mu, prior_logvar = model.class_prior(torch.tensor([0, 1]), n_tokens=2)

    assert prior_mu.shape == (2, 2, 4)
    assert prior_logvar.shape == (2, 2, 4)
    assert not torch.allclose(prior_mu[0], prior_mu[1])


def test_csvae_z_adversary_reads_the_complete_token_state():
    model = Model_CSVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        dim_w=4,
        n_head=1,
        factor=2,
    )

    first_linear = next(
        layer for layer in model.z_adversary if isinstance(layer, torch.nn.Linear)
    )

    assert first_linear.in_features == 2 * 4


def test_conditional_gaussian_kl_is_zero_for_matching_posteriors():
    prior_mu = torch.tensor([[[-1.0, 0.5]], [[1.0, -0.5]]])
    prior_logvar = torch.tensor([[[0.2, -0.1]], [[-0.3, 0.4]]])

    loss = conditional_gaussian_kl(
        prior_mu, prior_logvar, prior_mu, prior_logvar
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
