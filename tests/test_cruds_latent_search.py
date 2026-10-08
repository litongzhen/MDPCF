from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from mdpcf.sample import (
    expand_target_prior_for_search,
    initialize_latent_candidates,
    optimize_z_counterfactuals,
    optimize_w_counterfactuals,
    select_best_trajectory_snapshot,
    select_trajectory_quality_candidate_set,
    resolve_latent_initialization_center,
)


def test_trajectory_quality_score_uses_preregistered_weights_and_directions():
    snapshots = [
        {
            "diversity": 1.0,
            "lof": 3.0,
            "target_prob": 0.8,
            "proximity": 0.1,
        },
        {
            "diversity": 3.0,
            "lof": 1.0,
            "target_prob": 0.6,
            "proximity": 0.3,
        },
    ]

    best_index, scores = select_best_trajectory_snapshot(snapshots)

    assert best_index == 1
    assert scores == pytest.approx([0.3, 0.7])


def test_trajectory_snapshot_uses_lof_dpp_within_each_snapshot():
    info = {
        "idx_name_mapping": {"0": "x"},
        "num_col_idx": [0],
        "cat_col_idx": [],
        "num_min": {"x": 0.0},
        "num_max": {"x": 10.0},
    }
    snapshots = [
        {
            "step": 500,
            "cfs": pd.DataFrame({"x": [1.0, 9.0]}),
            "latents": torch.tensor([[1.0], [9.0]]),
            "decoded": torch.tensor([[1.0], [9.0]]),
            "target_probs": np.array([0.9, 0.8]),
            "lof_scores": np.array([10.0, 1.0]),
        }
    ]

    selected, diagnostics = select_trajectory_quality_candidate_set(
        snapshots, info, pd.Series({"x": 0.0}), total_CFs=1
    )

    assert selected["step"] == 500
    assert torch.equal(selected["latents"], torch.tensor([[9.0]]))
    assert diagnostics[0]["lof"] == pytest.approx(1.0)


def test_initial_candidates_lie_on_the_requested_w_radius():
    center = torch.zeros(4)
    candidates = initialize_latent_candidates(center, 8, 0.2, "cpu")

    distances = torch.linalg.vector_norm(candidates - center, dim=-1)

    assert torch.allclose(distances, torch.full_like(distances, 0.2), atol=1e-5)


def test_optimize_w_maps_public_w_z_order_to_the_shared_optimizer():
    w0 = torch.tensor([1.0])
    z0 = torch.tensor([2.0])
    with patch(
        "mdpcf.sample.optimize_z_counterfactuals", return_value=("w", "decoded")
    ) as shared:
        result = optimize_w_counterfactuals(w0, z0, "decoder", sentinel=True)

    assert result == ("w", "decoded")
    shared.assert_called_once_with(
        z0, w0, "decoder", sentinel=True, search_component="w"
    )


def test_target_prior_initialization_interpolates_from_source_w():
    source = torch.tensor([0.0, 2.0])
    target = torch.tensor([2.0, 0.0])

    center = resolve_latent_initialization_center(source, target, 0.5)

    assert torch.allclose(center, torch.tensor([1.0, 1.0]))


def test_token_target_prior_is_broadcast_to_every_feature_token():
    target_prior = torch.tensor([1.0, -1.0])

    expanded = expand_target_prior_for_search(target_prior, (3, 2))

    assert torch.equal(
        expanded,
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    )


def test_target_prior_rejects_incompatible_search_shape():
    with torch.no_grad(), pytest.raises(ValueError, match="target prior"):
        expand_target_prior_for_search(torch.ones(3), (2, 2))


def test_optimizer_returns_latent_and_decoding_from_the_same_best_step():
    class IdentityDecoder(torch.nn.Module):
        def set_fixed_w(self, _w):
            pass

        def clear_actionability_reference(self):
            pass

        def decode_z(self, candidates):
            return candidates.clone()

        def bb_only(self, decoded):
            return torch.stack((torch.zeros_like(decoded[:, 0]), decoded[:, 0]), dim=1)

    latent, decoded = optimize_z_counterfactuals(
        z0=torch.tensor([1.0]),
        w0=torch.tensor([0.0]),
        decoder_black_box_clf=IdentityDecoder(),
        total_CFs=1,
        min_iter=0,
        max_iter=1,
        proximity_weight_latent=0.0,
        proximity_weight_input=0.0,
        diversity_weight=0.0,
        latent_init_radius=0.0,
        target_class_index=1,
        original_input_instance=[1.0],
        immutable_mask=torch.tensor([False]),
        monotone_increase_mask=torch.tensor([False]),
        monotone_decrease_mask=torch.tensor([False]),
        proximity_input_loss="L2",
        continuous_indexes=[0],
        categorical_slices=[],
        num_feature_weights=torch.tensor([1.0]),
        validity_loss="cross_entropy",
        validity_margin=1.0,
        diversity_loss_type="average_distance",
        plausibility_weight_latent=0.0,
        plausibility_latent_loss="GMM",
        plausibility_model=None,
        device="cpu",
        enforce_constraints=False,
    )

    assert torch.allclose(latent, decoded)


def test_trajectory_variant_collects_snapshots_and_disables_early_stop():
    class IdentityDecoder(torch.nn.Module):
        def set_fixed_w(self, _w):
            pass

        def clear_actionability_reference(self):
            pass

        def decode_z(self, candidates):
            return candidates.clone()

        def bb_only(self, decoded):
            return torch.stack(
                (torch.zeros_like(decoded[:, 0]), decoded[:, 0]), dim=1
            )

    latent, decoded, snapshots = optimize_z_counterfactuals(
        z0=torch.tensor([1.0]),
        w0=torch.tensor([0.0]),
        decoder_black_box_clf=IdentityDecoder(),
        total_CFs=1,
        min_iter=0,
        max_iter=5,
        proximity_weight_latent=0.0,
        proximity_weight_input=0.0,
        diversity_weight=0.0,
        latent_init_radius=0.0,
        target_class_index=1,
        original_input_instance=[1.0],
        immutable_mask=torch.tensor([False]),
        monotone_increase_mask=torch.tensor([False]),
        monotone_decrease_mask=torch.tensor([False]),
        proximity_input_loss="L2",
        continuous_indexes=[0],
        categorical_slices=[],
        num_feature_weights=torch.tensor([1.0]),
        validity_loss="cross_entropy",
        validity_margin=1.0,
        diversity_loss_type="average_distance",
        plausibility_weight_latent=0.0,
        plausibility_latent_loss="GMM",
        plausibility_model=None,
        device="cpu",
        enforce_constraints=False,
        trajectory_quality_selection=True,
        trajectory_snapshot_interval=2,
    )

    assert torch.allclose(latent, decoded)
    assert [snapshot["step"] for snapshot in snapshots] == [0, 2, 4]
    assert all(
        torch.allclose(snapshot["latents"], snapshot["decoded"])
        for snapshot in snapshots
    )
