import unittest
from pathlib import Path
import sys

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdpcf.sample import (
    compute_auto_latent_init_radius,
    compute_dpp_diversity,
    compute_latent_gmm_plausibility_loss,
    resolve_latent_init_radius,
    select_diverse_counterfactual_subset,
)


class MDPCFDiversityLossTest(unittest.TestCase):
    def test_dpp_det_matches_negative_determinant_for_dice_style_loss(self):
        decoded = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32
        )
        pairwise_dist = torch.cdist(decoded, decoded, p=1)
        kernel = 1.0 / (1.0 + pairwise_dist)
        kernel = kernel + 1e-6 * torch.eye(kernel.shape[0])
        expected = -torch.linalg.det(kernel)

        actual = compute_dpp_diversity(decoded, variant="det")

        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_dpp_log_matches_negative_log_determinant(self):
        decoded = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32
        )
        pairwise_dist = torch.cdist(decoded, decoded, p=1)
        kernel = 1.0 / (1.0 + pairwise_dist)
        kernel = kernel + 1e-6 * torch.eye(kernel.shape[0])
        expected = -torch.logdet(kernel)

        actual = compute_dpp_diversity(decoded, variant="log")

        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_latent_gmm_plausibility_loss_prefers_high_density_latents(self):
        gmm_params = {
            "means": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "precisions": torch.eye(2, dtype=torch.float32).unsqueeze(0),
            "log_dets": torch.tensor([0.0], dtype=torch.float32),
            "log_weights": torch.tensor([0.0], dtype=torch.float32),
        }
        near = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
        far = torch.tensor([[3.0, 3.0]], dtype=torch.float32)

        near_loss = compute_latent_gmm_plausibility_loss(near, gmm_params)
        far_loss = compute_latent_gmm_plausibility_loss(far, gmm_params)

        self.assertLess(near_loss.item(), far_loss.item())

    def test_auto_latent_init_radius_uses_median_nearest_neighbor_distance(self):
        train_z = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]],
            dtype=torch.float32,
        )

        radius = compute_auto_latent_init_radius(train_z, scale=0.5)

        self.assertAlmostEqual(radius, 1.25)

    def test_resolve_latent_init_radius_keeps_fixed_value(self):
        class Args:
            latent_init_radius_strategy = "fixed"
            latent_init_radius = 0.84
            latent_init_radius_scale = 1.0
            latent_init_radius_max_samples = 5000

        radius = resolve_latent_init_radius(
            Args(), torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        )

        self.assertAlmostEqual(radius, 0.84)

    def test_lof_selection_prefers_low_plausibility_penalty(self):
        info = {
            "column_names": ["x", "target"],
            "target_col": "target",
            "idx_name_mapping": {"0": "x"},
            "num_col_idx": [0],
            "cat_col_idx": [],
            "num_min": {"x": 0.0},
            "num_max": {"x": 10.0},
        }
        cfs = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        latents = pd.DataFrame({"z_0": [0.0, 1.0, 2.0]})
        target_probs = [0.99, 0.7, 0.6]
        lof_scores = [10.0, 0.0, 1.0]

        selected_cfs, _, selected_probs = select_diverse_counterfactual_subset(
            cfs,
            latents,
            target_probs,
            info,
            total_CFs=2,
            selection_metric="lof_distance",
            candidate_plausibility_scores=lof_scores,
        )

        self.assertEqual(float(selected_cfs.iloc[0]["x"]), 5.0)
        self.assertEqual(float(selected_probs[0]), 0.7)


if __name__ == "__main__":
    unittest.main()
