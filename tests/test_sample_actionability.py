import unittest
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdpcf.sample import (
    Decoder_Black_Box,
    apply_numerical_domain_constraints,
    apply_raw_actionability_constraints,
    apply_dice_posthoc_sparsity,
    apply_posthoc_categorical_diversity,
    build_feature_change_summary,
    build_projection_shift_summary,
    build_sampling_diagnostics,
    encode_raw_features_for_black_box,
    select_diverse_counterfactual_subset,
    validate_counterfactuals,
    validate_counterfactual_constraints,
)


class _TwoBranchDecoder(torch.nn.Module):
    def forward(self, z, w):
        return z + 2.0 * w, []


class SampleActionabilityTest(unittest.TestCase):
    def test_decode_w_freezes_z_and_backpropagates_only_to_w(self):
        info = {
            "num_col_idx": [0, 1],
            "cat_col_idx": [],
            "idx_name_mapping": {"0": "a", "1": "b"},
            "column_domain": {},
            "ordinal": {},
            "ordinal_monotone_increase": [],
            "ordinal_monotone_decrease": [],
        }
        decoder = Decoder_Black_Box(
            torch.nn.Identity(), _TwoBranchDecoder(), info
        )
        z0 = torch.tensor([1.0, 2.0], requires_grad=True)
        w = torch.tensor([[3.0, 4.0]], requires_grad=True)

        decoder.set_fixed_z(z0)
        decoded = decoder.decode_w(w)
        decoded.sum().backward()

        self.assertIsNone(z0.grad)
        self.assertTrue(torch.allclose(w.grad, torch.full_like(w, 2.0)))

    def test_sample_pipeline_names_before_posthoc_output(self):
        sample_path = Path(__file__).resolve().parents[1] / "mdpcf" / "sample.py"
        content = sample_path.read_text(encoding="utf-8")

        self.assertIn("cfs_before_posthoc_sparsity.csv", content)

    def test_ordinal_monotone_increase_projects_lower_category_to_original(self):
        info = {
            "immutable": [],
            "monotone_increase": [],
            "monotone_decrease": [],
            "ordinal": {
                "education": [
                    " 11th",
                    " HS-grad",
                    " Bachelors",
                    " Masters",
                ]
            },
            "ordinal_monotone_increase": ["education"],
        }
        original = pd.Series({"education": " Bachelors"})
        cfs = pd.DataFrame({"education": [" 11th", " Masters"]})

        apply_raw_actionability_constraints(cfs, original, info)

        self.assertEqual(cfs["education"].tolist(), [" Bachelors", " Masters"])

    def test_ordinal_monotone_increase_masks_lower_category_logits(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [2],
            "idx_name_mapping": {"0": "age", "2": "education"},
            "column_domain": {
                "education": [
                    " 11th",
                    " HS-grad",
                    " Bachelors",
                    " Masters",
                ]
            },
            "ordinal": {
                "education": [
                    " 11th",
                    " HS-grad",
                    " Bachelors",
                    " Masters",
                ]
            },
            "ordinal_monotone_increase": ["education"],
        }
        decoder = Decoder_Black_Box(torch.nn.Identity(), torch.nn.Identity(), info)
        reference = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        decoder.set_actionability_reference(
            torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.bool),
            reference,
        )

        probs = decoder._process_cat_outputs([torch.zeros(1, 4)])[0]

        self.assertTrue(torch.allclose(probs[:, :2], torch.zeros(1, 2)))
        self.assertTrue(torch.all(probs[:, 2:] > 0))

    def test_numerical_domain_constraints_clip_all_numeric_columns(self):
        info = {
            "num_col_idx": [0, 8, 9, 10],
            "idx_name_mapping": {
                "0": "age",
                "8": "capital.gain",
                "9": "capital.loss",
                "10": "hours.per.week",
            },
            "column_domain": {
                "age": [17.0, 90.0],
                "capital.gain": [0.0, 99999.0],
                "capital.loss": [0.0, 4356.0],
                "hours.per.week": [1.0, 99.0],
            },
        }
        cfs = pd.DataFrame(
            {
                "age": [10.0, 100.0],
                "capital.gain": [-5.0, 120000.0],
                "capital.loss": [-1.0, 5000.0],
                "hours.per.week": [0.0, 120.0],
            }
        )

        apply_numerical_domain_constraints(cfs, info)

        self.assertEqual(cfs["age"].tolist(), [17.0, 90.0])
        self.assertEqual(cfs["capital.gain"].tolist(), [0.0, 99999.0])
        self.assertEqual(cfs["capital.loss"].tolist(), [0.0, 4356.0])
        self.assertEqual(cfs["hours.per.week"].tolist(), [1.0, 99.0])

    def test_validate_counterfactuals_filters_all_final_constraints(self):
        info = {
            "immutable": ["sex"],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "integer": ["age"],
            "num_col_idx": [0, 10],
            "cat_col_idx": [2, 7],
            "idx_name_mapping": {
                "0": "age",
                "2": "education",
                "7": "sex",
                "10": "hours.per.week",
            },
            "column_domain": {
                "age": [17.0, 90.0],
                "hours.per.week": [1.0, 99.0],
                "education": [" 11th", " HS-grad", " Bachelors"],
                "sex": [" Male", " Female"],
            },
            "ordinal": {
                "education": [" 11th", " HS-grad", " Bachelors"],
            },
            "ordinal_monotone_increase": ["education"],
            "ordinal_monotone_decrease": [],
        }
        original = pd.Series(
            {
                "age": 30.0,
                "hours.per.week": 40.0,
                "education": " HS-grad",
                "sex": " Male",
            }
        )
        cfs = pd.DataFrame(
            [
                {
                    "age": 31.0,
                    "hours.per.week": 45.0,
                    "education": " Bachelors",
                    "sex": " Male",
                },
                {
                    "age": 29.0,
                    "hours.per.week": 45.0,
                    "education": " Bachelors",
                    "sex": " Male",
                },
                {
                    "age": 31.5,
                    "hours.per.week": 45.0,
                    "education": " Bachelors",
                    "sex": " Male",
                },
                {
                    "age": 31.0,
                    "hours.per.week": 100.0,
                    "education": " Bachelors",
                    "sex": " Male",
                },
                {
                    "age": 31.0,
                    "hours.per.week": 45.0,
                    "education": " 11th",
                    "sex": " Male",
                },
                {
                    "age": 31.0,
                    "hours.per.week": 45.0,
                    "education": " Bachelors",
                    "sex": " Female",
                },
                {
                    "age": 31.0,
                    "hours.per.week": 45.0,
                    "education": " Bachelors",
                    "sex": " Male",
                },
            ]
        )
        target_probs = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4])

        mask = validate_counterfactuals(cfs, original, info, target_probs)

        self.assertEqual(mask.tolist(), [True, False, False, False, False, False, False])

    def test_validate_counterfactuals_can_ignore_final_constraints(self):
        info = {
            "immutable": ["sex"],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "integer": ["age"],
            "num_col_idx": [0],
            "cat_col_idx": [7],
            "idx_name_mapping": {"0": "age", "7": "sex"},
            "column_domain": {"age": [17.0, 90.0], "sex": [" Male", " Female"]},
        }
        original = pd.Series({"age": 30.0, "sex": " Male"})
        cfs = pd.DataFrame(
            [
                {"age": 29.0, "sex": " Female"},
                {"age": 31.5, "sex": " Male"},
                {"age": 31.0, "sex": " Male"},
            ]
        )
        target_probs = torch.tensor([0.9, 0.9, 0.4])

        mask = validate_counterfactuals(
            cfs, original, info, target_probs, enforce_constraints=False
        )

        self.assertEqual(mask.tolist(), [True, True, False])

    def test_validate_counterfactual_constraints_excludes_target_probability(self):
        info = {
            "immutable": ["sex"],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "integer": ["age"],
            "num_col_idx": [0],
            "cat_col_idx": [7],
            "idx_name_mapping": {"0": "age", "7": "sex"},
            "column_domain": {"age": [17.0, 90.0], "sex": [" Male", " Female"]},
        }
        original = pd.Series({"age": 30.0, "sex": " Male"})
        cfs = pd.DataFrame(
            [
                {"age": 31.0, "sex": " Male"},
                {"age": 29.0, "sex": " Male"},
                {"age": 31.0, "sex": " Female"},
            ]
        )

        mask = validate_counterfactual_constraints(cfs, original, info)

        self.assertEqual(mask.tolist(), [True, False, False])

    def test_build_sampling_diagnostics_computes_filter_reasons(self):
        stats = {
            "attempted_count": 10,
            "optimized_valid_source_count": 4,
            "optimized_valid_candidate_count": 6,
            "post_projection_candidate_count": 6,
            "post_projection_validity_count": 3,
            "post_projection_constraints_count": 4,
            "saved_cf_count": 2,
            "saved_source_count": 2,
        }

        diagnostics = build_sampling_diagnostics(stats)
        row = diagnostics.iloc[0]

        self.assertEqual(row["optimization_failed_count"], 6)
        self.assertEqual(row["filtered_by_validity"], 3)
        self.assertEqual(row["filtered_by_constraints"], 2)
        self.assertEqual(row["coverage"], 0.2)

    def test_select_diverse_counterfactual_subset_uses_mixed_distance(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "num_min": {"age": 20.0},
            "num_max": {"age": 60.0},
        }
        cfs = pd.DataFrame(
            [
                {"age": 31.0, "education": " HS-grad"},
                {"age": 32.0, "education": " HS-grad"},
                {"age": 55.0, "education": " Masters"},
                {"age": 50.0, "education": " Bachelors"},
            ]
        )
        latents = pd.DataFrame({"z_0": [0.0, 1.0, 2.0, 3.0]})
        target_probs = [0.95, 0.90, 0.70, 0.80]

        selected_cfs, selected_latents, selected_probs = select_diverse_counterfactual_subset(
            cfs, latents, target_probs, info, total_CFs=3, selection_metric="mixed_distance"
        )

        self.assertEqual(selected_cfs.index.tolist(), [0, 2, 3])
        self.assertEqual(selected_latents["z_0"].tolist(), [0.0, 2.0, 3.0])
        self.assertEqual(selected_probs.tolist(), [0.95, 0.70, 0.80])

    def test_select_diverse_counterfactual_subset_can_prioritize_categorical_diversity(self):
        info = {
            "num_col_idx": [0, 1, 2],
            "cat_col_idx": [3],
            "idx_name_mapping": {
                "0": "age",
                "1": "hours.per.week",
                "2": "capital.gain",
                "3": "education",
            },
            "num_min": {"age": 20.0},
            "num_max": {"age": 60.0},
        }
        cfs = pd.DataFrame(
            [
                {
                    "age": 30.0,
                    "hours.per.week": 0.0,
                    "capital.gain": 0.0,
                    "education": " HS-grad",
                },
                {
                    "age": 60.0,
                    "hours.per.week": 1.0,
                    "capital.gain": 1.0,
                    "education": " HS-grad",
                },
                {
                    "age": 31.0,
                    "hours.per.week": 0.0,
                    "capital.gain": 0.0,
                    "education": " Masters",
                },
            ]
        )
        latents = pd.DataFrame({"z_0": [0.0, 1.0, 2.0]})
        target_probs = [0.95, 0.90, 0.80]

        selected_cfs, _, _ = select_diverse_counterfactual_subset(
            cfs,
            latents,
            target_probs,
            info,
            total_CFs=2,
            selection_metric="category_first",
        )

        self.assertEqual(selected_cfs.index.tolist(), [0, 2])

    def test_build_projection_shift_summary_measures_landing_shift(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [2],
            "idx_name_mapping": {"0": "age", "2": "education"},
            "column_domain": {"education": [" HS-grad", " Bachelors"]},
        }
        decoded = torch.tensor([[0.2, 0.7, 0.3], [0.4, 0.2, 0.8]])
        rounded = torch.tensor([[0.3, 0.0, 1.0], [0.4, 0.0, 1.0]])
        before_probs = torch.tensor([0.9, 0.6])
        after_probs = torch.tensor([0.4, 0.7])
        validity_mask = [False, True]
        constraints_mask = [True, True]

        summary = build_projection_shift_summary(
            source_id=5,
            decoded_encoded=decoded,
            rounded_encoded=rounded,
            before_probs=before_probs,
            after_probs=after_probs,
            validity_mask=validity_mask,
            constraints_mask=constraints_mask,
            info=info,
        )

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary.loc[0, "id"], 5)
        self.assertEqual(summary.loc[0, "candidate_index"], 0)
        self.assertAlmostEqual(summary.loc[0, "prob_drop"], 0.5)
        self.assertAlmostEqual(summary.loc[0, "encoded_l1_shift"], 1.5)
        self.assertFalse(bool(summary.loc[0, "saved"]))
        self.assertTrue(bool(summary.loc[1, "saved"]))

    def test_encode_raw_features_uses_model_category_order_when_available(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [7],
            "idx_name_mapping": {"0": "age", "7": "sex"},
            "num_min": {"age": "0.0"},
            "num_max": {"age": "100.0"},
            "column_domain": {"sex": [" Male", " Female"]},
            "cat_value_orders": {"sex": [" Female", " Male"]},
        }
        df = pd.DataFrame({"age": [50.0], "sex": [" Male"]})

        encoded = encode_raw_features_for_black_box(df, info)

        self.assertEqual(encoded.tolist(), [[0.5, 0.0, 1.0]])

    def test_build_feature_change_summary_is_dataset_agnostic(self):
        info = {
            "target_col": "income",
            "num_col_idx": [0, 10],
            "cat_col_idx": [2, 7],
            "idx_name_mapping": {
                "0": "age",
                "2": "education",
                "7": "sex",
                "10": "hours.per.week",
            },
        }
        originals = pd.DataFrame(
            {
                "id": [0, 1],
                "age": [30.0, 40.0],
                "hours.per.week": [40.0, 20.0],
                "education": [" HS-grad", " Bachelors"],
                "sex": [" Male", " Female"],
            }
        )
        cfs = pd.DataFrame(
            {
                "id": [0, 1],
                "age": [31.0, 40.0],
                "hours.per.week": [45.0, 10.0],
                "education": [" Bachelors", " Bachelors"],
                "sex": [" Male", " Male"],
                "target_prob": [0.8, 0.9],
            }
        )

        summary = build_feature_change_summary(cfs, originals, info)
        by_feature = summary.set_index("feature")

        self.assertEqual(by_feature.loc["age", "feature_type"], "numerical")
        self.assertEqual(by_feature.loc["age", "changed_count"], 1)
        self.assertEqual(by_feature.loc["age", "changed_rate"], 0.5)
        self.assertEqual(by_feature.loc["age", "mean_abs_change"], 0.5)
        self.assertEqual(by_feature.loc["hours.per.week", "max_abs_change"], 10.0)
        self.assertEqual(by_feature.loc["education", "feature_type"], "categorical")
        self.assertEqual(by_feature.loc["education", "top_transition"], " HS-grad ->  Bachelors")
        self.assertEqual(by_feature.loc["sex", "changed_count"], 1)

    def test_dice_posthoc_sparsity_only_moves_continuous_features_toward_original(self):
        info = {
            "num_col_idx": [0, 10],
            "cat_col_idx": [2],
            "idx_name_mapping": {
                "0": "age",
                "2": "education",
                "10": "hours.per.week",
            },
            "integer": ["age"],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "column_domain": {
                "age": [17.0, 90.0],
                "hours.per.week": [1.0, 99.0],
                "education": [" HS-grad", " Bachelors"],
            },
        }
        original = pd.Series(
            {"age": 30, "hours.per.week": 40.0, "education": " HS-grad"}
        )
        cfs = pd.DataFrame(
            [
                {"age": 35, "hours.per.week": 50.0, "education": " Bachelors"},
            ]
        )

        def predict_target_prob(frame):
            valid_hours = frame["hours.per.week"].astype(float).to_numpy() >= 45.0
            return valid_hours.astype(float)

        sparse_cfs, probs, summary = apply_dice_posthoc_sparsity(
            cfs, original, info, predict_target_prob
        )

        self.assertEqual(sparse_cfs.loc[0, "age"], 30)
        self.assertGreaterEqual(sparse_cfs.loc[0, "hours.per.week"], 45.0)
        self.assertLess(sparse_cfs.loc[0, "hours.per.week"], 50.0)
        self.assertEqual(sparse_cfs.loc[0, "education"], " Bachelors")
        self.assertEqual(probs.tolist(), [1.0])
        self.assertEqual(summary.loc[0, "continuous_revert_attempted"], 2)
        self.assertEqual(summary.loc[0, "continuous_revert_accepted"], 2)
        self.assertEqual(summary.loc[0, "changed_features_before"], 3)
        self.assertEqual(summary.loc[0, "changed_features_after"], 2)

    def test_dice_posthoc_sparsity_can_ignore_constraints(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "immutable": [],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "integer": ["age"],
            "column_domain": {"age": [17.0, 90.0], "education": [" HS-grad"]},
        }
        original = pd.Series({"age": 30, "education": " HS-grad"})
        cfs = pd.DataFrame([{"age": 29.0, "education": " HS-grad"}])

        def predict_target_prob(frame):
            return (frame["age"].astype(float).to_numpy() < 30.0).astype(float)

        sparse_cfs, probs, _ = apply_dice_posthoc_sparsity(
            cfs,
            original,
            info,
            predict_target_prob,
            threshold=0.5,
            enforce_constraints=False,
        )

        self.assertLess(float(sparse_cfs.loc[0, "age"]), 30.0)
        self.assertEqual(probs.tolist(), [1.0])

    def test_dice_posthoc_sparsity_binary_moves_continuous_feature_when_full_revert_fails(self):
        info = {
            "num_col_idx": [10],
            "cat_col_idx": [],
            "idx_name_mapping": {"10": "hours.per.week"},
            "integer": [],
            "monotone_increase": [],
            "monotone_decrease": [],
            "column_domain": {"hours.per.week": [1.0, 99.0]},
        }
        original = pd.Series({"hours.per.week": 40.0})
        cfs = pd.DataFrame([{"hours.per.week": 50.0}])

        def predict_target_prob(frame):
            valid = frame["hours.per.week"].astype(float).to_numpy() >= 45.0
            return valid.astype(float)

        sparse_cfs, probs, summary = apply_dice_posthoc_sparsity(
            cfs, original, info, predict_target_prob
        )

        self.assertGreaterEqual(sparse_cfs.loc[0, "hours.per.week"], 45.0)
        self.assertLess(sparse_cfs.loc[0, "hours.per.week"], 50.0)
        self.assertEqual(probs.tolist(), [1.0])
        self.assertEqual(summary.loc[0, "continuous_revert_accepted"], 1)

    def test_posthoc_categorical_diversity_accepts_only_diversity_improving_valid_changes(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "num_min": {"age": 20.0},
            "num_max": {"age": 60.0},
            "immutable": [],
            "monotone_increase": [],
            "monotone_decrease": [],
            "integer": ["age"],
            "column_domain": {
                "age": [20.0, 60.0],
                "education": [" HS-grad", " Bachelors", " Masters"],
            },
        }
        original = pd.Series({"age": 30.0, "education": " HS-grad"})
        cfs = pd.DataFrame(
            [
                {"age": 31.0, "education": " Bachelors"},
                {"age": 32.0, "education": " Bachelors"},
            ]
        )

        def predict_target_prob(frame):
            return (frame["education"] != " HS-grad").astype(float).to_numpy()

        diverse_cfs, probs, summary = apply_posthoc_categorical_diversity(
            cfs, original, info, predict_target_prob, threshold=0.5
        )

        self.assertEqual(sorted(diverse_cfs["education"].tolist()), [" Bachelors", " Masters"])
        self.assertEqual(probs.tolist(), [1.0, 1.0])
        self.assertGreaterEqual(int(summary["categorical_diversity_accepted"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
