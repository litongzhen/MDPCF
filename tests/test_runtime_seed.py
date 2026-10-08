import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils import get_args, set_global_seed
from utils_train import get_black_box_model_path, get_name_form_args, get_optim_params_form_args
from mdpcf.csvae.main import (
    checkpoint_artifacts_ready,
    encode_label_related_latents,
    latent_artifact_metadata,
)
from mdpcf.latent_utils import load_test_latent_arrays
from mdpcf.csvae.model import Model_StandardVAE


def test_set_global_seed_reproduces_python_numpy_and_torch_draws():
    set_global_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_global_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_run_tag_is_added_to_seeded_model_artifact_names():
    args = SimpleNamespace(
        num_encoding="minmax",
        latent_model="csvae",
        csvae_backbone="transformer",
        num_reconstr_loss="L2",
        kl_weight=-1,
        gumbel_softmax=True,
        tau=1.0,
        sigmoid=True,
        run_tag="multiseed",
        seed=43,
        hidden_dims=16,
    )

    assert get_name_form_args(args).endswith("run[multiseed]_seed[43]")
    assert "run[multiseed]_seed[43]" in get_black_box_model_path(args, "data/adult")


def test_nondefault_latent_dimensions_are_isolated_in_artifact_name():
    base = dict(
        num_encoding="minmax",
        latent_model="csvae",
        csvae_backbone="transformer",
        num_reconstr_loss="L2",
        kl_weight=-1,
        gumbel_softmax=True,
        tau=1.0,
        sigmoid=True,
        run_tag="d6_probe",
        seed=42,
    )

    d4_name = get_name_form_args(
        SimpleNamespace(**base, token_dim=4, latent_dim=4)
    )
    d6_name = get_name_form_args(
        SimpleNamespace(**base, token_dim=6, latent_dim=6)
    )

    assert "dim[" not in d4_name
    assert "dim[t6_z6]" in d6_name


def test_dimension_cli_defaults_to_d4_and_accepts_d6(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    defaults = get_args()
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--token_dim", "6", "--latent_dim", "6"]
    )
    d6 = get_args()

    assert (defaults.token_dim, defaults.latent_dim) == (4, 4)
    assert (d6.token_dim, d6.latent_dim) == (6, 6)


def test_trajectory_quality_selection_is_explicit_and_disabled_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    defaults = get_args()
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--trajectory_quality_selection", "True"]
    )
    enabled = get_args()

    assert defaults.trajectory_quality_selection is False
    assert enabled.trajectory_quality_selection is True


def test_black_box_seed_can_be_fixed_independently_of_generation_seed():
    args = SimpleNamespace(
        hidden_dims=16,
        gumbel_softmax=True,
        run_tag="multiseed",
        seed=44,
        black_box_seed=42,
    )

    assert get_black_box_model_path(args, "data/adult").endswith(
        "black_box_mlp_hidden_16_run[multiseed]_seed[42].pkl"
    )


def test_seeded_run_uses_a_short_result_directory_name():
    args = SimpleNamespace(
        method="mdpcf",
        run_tag="multiseed",
        seed=43,
        proximity_weight_input=1.0,
        diversity_weight=1.0,
        diversity_loss="average_distance",
        latent_init_radius=0.5,
        latent_init_radius_strategy="auto_nn",
        latent_init_radius_scale=0.2,
        hidden_dims=16,
        proximity_weight_latent=0.1,
        proximity_latent_loss="L2",
        plausibility_weight_latent=-1.0,
        plausibility_latent_loss="GMM",
        proximity_input_loss="reconstruction",
        validity_loss="hinge",
        validity_margin=1.0,
        categorical_straight_through=True,
        posthoc_sparsity=False,
        posthoc_diversity=False,
        candidate_CFs=30,
        total_CFs=10,
        cf_selection_metric="category_first",
        enforce_constraints=False,
        immutable=False,
    )

    assert get_optim_params_form_args(args) == "run[multiseed]_seed[43]"


def test_multiseed_runner_defines_seeded_five_dataset_sweep():
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_final_main_ablations.ps1"

    assert runner.exists()
    content = runner.read_text(encoding="utf-8")
    assert '"adult", "bank", "default", "gmc", "lending-club"' in content
    assert '[int[]]$Seeds = @(42, 43, 44)' in content
    assert '"csvae_transformer", "csvae_mlp", "standard_vae"' in content
    assert '[string]$RunTag = "final_main_ablation_e2000"' in content
    assert '[int]$BlackBoxSeed = 42' in content
    assert '"--black_box_seed", "$BlackBoxSeed"' in content
    assert '[int]$CSVAEpochs = 2000' in content
    assert '"--validation_fraction", "0.1"' in content
    assert '"--mlp_parameter_match", "True"' in content
    assert '"adult" { return 1 }' in content
    assert '"default" { return 0 }' in content
    assert '"gmc" { return 0 }' in content
    assert '"--candidate_CFs", "50"' in content
    assert '"--w_structure", "token"' in content
    assert '"--w_dim", "4"' in content
    assert '"--source_start", "0"' in content
    assert '[switch]$Resume' in content
    assert 'Test-SeededEvaluationComplete' in content


def test_multiseed_runner_accepts_an_isolated_results_root():
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_final_main_ablations.ps1"

    content = runner.read_text(encoding="utf-8")

    assert '[string]$ResultsRoot' in content
    assert 'results\\final_main_ablation_3000\\runs' in content


def test_checkpoint_artifacts_ready_requires_all_sampling_dependencies(tmp_path):
    assert not checkpoint_artifacts_ready(tmp_path)

    for name in [
        "model.pt",
        "encoder.pt",
        "decoder.pt",
        "train_z.npy",
        "test_z.npy",
        "train_w.npy",
        "test_w.npy",
    ]:
        (tmp_path / name).touch()
    (tmp_path / "latent_artifact_protocol.json").write_text(
        json.dumps(latent_artifact_metadata("csvae")), encoding="utf-8"
    )

    assert checkpoint_artifacts_ready(tmp_path)


def test_standard_vae_checkpoint_does_not_require_residual_arrays(tmp_path):
    for name in ["model.pt", "encoder.pt", "decoder.pt", "train_z.npy", "test_z.npy"]:
        (tmp_path / name).touch()
    (tmp_path / "latent_artifact_protocol.json").write_text(
        json.dumps(latent_artifact_metadata("standard_vae")), encoding="utf-8"
    )

    assert checkpoint_artifacts_ready(tmp_path, latent_model="standard_vae")


def test_checkpoint_artifacts_ready_rejects_a_stale_protocol_marker(tmp_path):
    for name in [
        "model.pt",
        "encoder.pt",
        "decoder.pt",
        "train_z.npy",
        "test_z.npy",
        "train_w.npy",
        "test_w.npy",
    ]:
        (tmp_path / name).touch()
    (tmp_path / "latent_artifact_protocol.json").write_text(
        json.dumps({"latent_artifact_protocol": "source_residual_v2"}),
        encoding="utf-8",
    )

    assert not checkpoint_artifacts_ready(tmp_path)


def test_label_related_latents_use_explicit_inference_labels():
    class RecordingEncoder:
        def __init__(self):
            self.received_y = object()

        def __call__(self, x_num, x_cat, y=None, deterministic=False):
            self.received_y = y
            return (None, None, None, None, None, torch.ones(2, 3, 4), None)

    encoder = RecordingEncoder()
    labels = torch.tensor([0, 1])
    latents = encode_label_related_latents(
        encoder,
        torch.zeros(2, 3),
        torch.zeros(2, 1, dtype=torch.long),
        labels,
    )

    assert torch.equal(encoder.received_y, labels)
    assert latents.shape == (2, 3, 4)


def test_csvae_metadata_declares_cruds_aligned_latent_roles():
    metadata = latent_artifact_metadata("csvae", w_structure="token", w_dim=4)

    assert metadata["latent_artifact_protocol"] == "cruds_aligned_v3"
    assert metadata["search_latent"] == "w"
    assert metadata["fixed_latent"] == "z"
    assert metadata["w_structure"] == "token"
    assert metadata["w_dim"] == 4


def test_d6_metadata_records_model_dimensions():
    metadata = latent_artifact_metadata(
        "csvae", w_structure="token", w_dim=6, token_dim=6, latent_dim=6
    )

    assert metadata["token_dim"] == 6
    assert metadata["latent_dim"] == 6


def test_checkpoint_reuse_rejects_a_different_w_structure(tmp_path):
    for name in [
        "model.pt",
        "encoder.pt",
        "decoder.pt",
        "train_z.npy",
        "test_z.npy",
        "train_w.npy",
        "test_w.npy",
    ]:
        (tmp_path / name).touch()
    (tmp_path / "latent_artifact_protocol.json").write_text(
        json.dumps(latent_artifact_metadata("csvae", "global", 2)),
        encoding="utf-8",
    )

    assert not checkpoint_artifacts_ready(
        tmp_path, latent_model="csvae", w_structure="token", w_dim=4
    )


def test_loading_latents_rejects_an_unmarked_cruds_protocol(tmp_path):
    np.save(tmp_path / "test_z.npy", np.zeros((2, 3, 4), dtype=np.float32))
    np.save(tmp_path / "test_w.npy", np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(RuntimeError, match="cruds_aligned_v3"):
        load_test_latent_arrays(tmp_path)


def test_loading_latents_rejects_old_source_residual_protocol(tmp_path):
    np.save(tmp_path / "test_z.npy", np.zeros((2, 3, 4), dtype=np.float32))
    np.save(tmp_path / "test_w.npy", np.zeros((2, 3, 4), dtype=np.float32))
    (tmp_path / "latent_artifact_protocol.json").write_text(
        '{"latent_artifact_protocol": "source_residual_v2"}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="cruds_aligned_v3"):
        load_test_latent_arrays(tmp_path)


def test_standard_vae_has_only_a_single_latent_branch():
    model = Model_StandardVAE(
        num_layers=1,
        d_numerical=1,
        categories=[2],
        d_token=4,
        dim_z=4,
        n_head=1,
        factor=2,
    )

    outputs = model(
        torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.long)
    )

    assert len(outputs) == 5
    assert not any(
        "w_" in name or name.endswith("y_head")
        for name, _ in model.named_parameters()
    )
