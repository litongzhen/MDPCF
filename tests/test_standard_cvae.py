import json
import pytest
import torch

from utils import get_args
from mdpcf.csvae.main import (
    encode_conditional_latents,
    latent_artifact_metadata,
    validation_objective,
)
from mdpcf.latent_utils import validate_latent_artifact_protocol
from mdpcf.sample import resolve_fixed_decoder_context
from mdpcf.csvae.model import (
    ConditionalVAEDecoderModel,
    ConditionalVAEEncoderModel,
    Model_ConditionalVAE,
)


def _make_model():
    return Model_ConditionalVAE(
        num_layers=1,
        d_numerical=2,
        categories=[3],
        d_token=4,
        dim_z=4,
        n_head=1,
        factor=2,
        gumbel_softmax=False,
        sigmoid=True,
    ).eval()


def test_cvae_deterministic_encoder_depends_on_binary_condition():
    torch.manual_seed(42)
    model = _make_model()
    x_num = torch.tensor([[0.2, 0.8]])
    x_cat = torch.tensor([[1]])

    _, _, z0 = model.encoder(x_num, x_cat, torch.tensor([0]), deterministic=True)
    _, _, z1 = model.encoder(x_num, x_cat, torch.tensor([1]), deterministic=True)

    assert z0.shape == (1, 3, 4)
    assert not torch.allclose(z0, z1)


def test_cvae_deterministic_decoder_depends_on_binary_condition():
    torch.manual_seed(42)
    model = _make_model()
    z = torch.zeros(1, 3, 4)

    num0, cat0 = model.decoder(z, torch.tensor([0]), deterministic=True)
    num1, cat1 = model.decoder(z, torch.tensor([1]), deterministic=True)

    assert num0.shape == (1, 2)
    assert len(cat0) == 1
    assert cat0[0].shape == (1, 3)
    assert not torch.allclose(num0, num1) or not torch.allclose(cat0[0], cat1[0])


def test_cvae_sampling_wrappers_preserve_condition_interface():
    torch.manual_seed(42)
    model = _make_model()
    encoder = ConditionalVAEEncoderModel(
        1, 2, [3], 4, n_head=1, factor=2
    )
    decoder = ConditionalVAEDecoderModel(
        1, 2, [3], 4, n_head=1, factor=2, sigmoid=True
    )
    encoder.load_weights(model)
    decoder.load_weights(model)

    x_num = torch.tensor([[0.2, 0.8]])
    x_cat = torch.tensor([[1]])
    z = encoder(x_num, x_cat, torch.tensor([1]))
    reconstructed = decoder(z, torch.tensor([1]))

    assert z.shape == (1, 3, 4)
    assert reconstructed[0].shape == (1, 2)


def test_cvae_cli_is_an_explicit_latent_model_choice(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--latent_model", "cvae"])

    assert get_args().latent_model == "cvae"


def test_cvae_validation_objective_excludes_csvae_only_terms():
    result = validation_objective(
        "cvae",
        mse_loss=torch.tensor(1.0),
        ce_loss=torch.tensor(2.0),
        kl_z_loss=torch.tensor(3.0),
        kl_w_loss=torch.tensor(100.0),
        y_loss=torch.tensor(100.0),
        beta=0.1,
    )

    assert result.item() == pytest.approx(3.3)


def test_cvae_artifact_protocol_is_isolated_and_validated(tmp_path):
    metadata = latent_artifact_metadata("cvae")

    assert metadata == {
        "latent_artifact_protocol": "conditional_vae_v1",
        "source_label_mode": "black_box_prediction",
        "search_latent": "z",
        "fixed_condition": "target_class",
    }
    (tmp_path / "latent_artifact_protocol.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    validate_latent_artifact_protocol(tmp_path, latent_model="cvae")


def test_cvae_latent_encoding_uses_supplied_black_box_labels():
    class CapturingEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.labels = None

        def forward(self, x_num, x_cat, labels, deterministic=False):
            self.labels = labels.detach().clone()
            z = labels.float().reshape(-1, 1, 1)
            return z, z, z

    encoder = CapturingEncoder()
    labels = torch.tensor([1, 0])

    encoded = encode_conditional_latents(
        encoder, torch.zeros(2, 1), torch.zeros(2, 0).long(), labels
    )

    assert torch.equal(encoder.labels, labels)
    assert torch.equal(encoded.reshape(-1), labels.float())


def test_cvae_sampling_fixes_decoder_to_target_condition():
    source_context = torch.tensor([0.0, 0.0, 0.0, 0.0])

    cvae_context = resolve_fixed_decoder_context(
        "cvae", source_context, target_class_index=1, device="cpu"
    )
    standard_context = resolve_fixed_decoder_context(
        "standard_vae", source_context, target_class_index=1, device="cpu"
    )

    assert torch.equal(cvae_context, torch.tensor([1]))
    assert torch.equal(standard_context, source_context)
