import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_framework.evaluate import (
    build_evaluation_summary,
    build_feature_change_summary,
    compute_lof_log_scale_metrics,
    compute_nearest_neighbor_distance,
)


class EvaluationSummaryTest(unittest.TestCase):
    def test_compute_lof_log_scale_metrics_uses_valid_counterfactuals(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "target_col": "income",
            "column_names": ["age", "education", "income"],
        }
        train_df = pd.DataFrame(
            [
                {"age": 25, "education": " HS-grad", "income": 0},
                {"age": 30, "education": " HS-grad", "income": 0},
                {"age": 35, "education": " Bachelors", "income": 1},
                {"age": 40, "education": " Bachelors", "income": 1},
            ]
        )
        originals_by_id = {
            0: {"id": 0, "age": 30, "education": " HS-grad"},
            1: {"id": 1, "age": 35, "education": " Bachelors"},
        }
        valid_cfs = pd.DataFrame(
            [
                {"id": 0, "age": 31, "education": " Bachelors", "target_prob": 0.8},
                {"id": 1, "age": 36, "education": " Bachelors", "target_prob": 0.9},
            ]
        )

        metrics = compute_lof_log_scale_metrics(
            valid_cfs, originals_by_id, train_df, info, n_neighbors=2
        )

        self.assertEqual(metrics["lof_count"], 2)
        self.assertTrue(np.isfinite(metrics["lof_log_scale"]))
        self.assertEqual(metrics["lof_log_scale"], metrics["lof_score_cf"])
        self.assertTrue(np.isfinite(metrics["lof_score_original"]))

    def test_compute_nearest_neighbor_distance_uses_l2_distance(self):
        query_latents = np.array([[1.0, 1.0], [4.0, 0.0]])
        reference_latents = np.array([[0.0, 0.0], [1.0, 3.0], [5.0, 0.0]])

        metrics = compute_nearest_neighbor_distance(query_latents, reference_latents)

        self.assertEqual(metrics["latent_nn_count"], 2)
        self.assertAlmostEqual(metrics["latent_nn_distance"], (np.sqrt(2.0) + 1.0) / 2)

    def test_build_evaluation_summary_reports_core_metrics(self):
        info = {
            "num_col_idx": [0, 1],
            "cat_col_idx": [2],
            "idx_name_mapping": {
                "0": "age",
                "1": "hours.per.week",
                "2": "education",
            },
            "target_col": "income",
            "column_names": ["age", "hours.per.week", "education", "income"],
            "integer": ["age"],
            "immutable": [],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "column_domain": {
                "age": [17.0, 90.0],
                "hours.per.week": [1.0, 99.0],
                "education": [" HS-grad", " Bachelors"],
            },
        }
        originals = pd.DataFrame(
            [
                {"id": 0, "age": 30, "hours.per.week": 40, "education": " HS-grad"},
                {"id": 1, "age": 50, "hours.per.week": 30, "education": " Bachelors"},
            ]
        )
        cfs = pd.DataFrame(
            [
                {
                    "id": 0,
                    "age": 31,
                    "hours.per.week": 40,
                    "education": " Bachelors",
                    "target_prob": 0.8,
                },
                {
                    "id": 0,
                    "age": 29,
                    "hours.per.week": 41,
                    "education": " HS-grad",
                    "target_prob": 0.9,
                },
                {
                    "id": 1,
                    "age": 50,
                    "hours.per.week": 30,
                    "education": " Bachelors",
                    "target_prob": 0.4,
                },
            ]
        )

        summary = build_evaluation_summary(cfs, originals, info, total_CFs=2)
        row = summary.iloc[0]

        self.assertEqual(row["attempted_count"], 2)
        self.assertEqual(row["generated_cf_count"], 3)
        self.assertEqual(row["valid_cf_count"], 2)
        self.assertEqual(row["valid_source_count"], 1)
        self.assertAlmostEqual(row["coverage"], 0.5)
        self.assertAlmostEqual(row["validity"], 2 / 3)
        self.assertAlmostEqual(row["validity_actual"], 2 / 3)
        self.assertAlmostEqual(row["validity_generated"], 2 / 3)
        self.assertAlmostEqual(row["validity_expected"], 0.5)
        self.assertAlmostEqual(row["mean_target_prob"], 0.85)
        self.assertAlmostEqual(row["sparsity"], 2 / 3)
        self.assertAlmostEqual(row["sparsity_cont"], 0.75)
        self.assertAlmostEqual(row["sparsity_cat"], 0.5)
        self.assertAlmostEqual(row["proximity"], 2 / 3)
        self.assertAlmostEqual(row["constraint_violation_rate"], 1 / 3)
        self.assertAlmostEqual(row["valid_constraint_violation_rate"], 0.5)
        self.assertEqual(row["feasible_cf_count"], 1)
        self.assertEqual(row["feasible_source_count"], 1)
        self.assertAlmostEqual(row["feasible_coverage"], 0.5)
        self.assertAlmostEqual(row["feasible_validity"], 1 / 3)
        self.assertAlmostEqual(row["feasible_validity_actual"], 1 / 3)
        self.assertAlmostEqual(row["feasible_validity_generated"], 1 / 3)
        self.assertAlmostEqual(row["feasible_validity_expected"], 0.25)
        self.assertAlmostEqual(row["feasible_mean_target_prob"], 0.8)

    def test_build_evaluation_summary_reports_prediction_and_feasible_diversity(self):
        info = {
            "num_col_idx": [0, 1],
            "cat_col_idx": [2],
            "idx_name_mapping": {
                "0": "age",
                "1": "hours.per.week",
                "2": "education",
            },
            "target_col": "income",
            "column_names": ["age", "hours.per.week", "education", "income"],
            "integer": ["age"],
            "immutable": [],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "num_min": {"age": 20.0, "hours.per.week": 0.0},
            "num_max": {"age": 40.0, "hours.per.week": 50.0},
            "column_domain": {
                "age": [20.0, 40.0],
                "hours.per.week": [0.0, 50.0],
                "education": [" HS-grad", " Bachelors"],
            },
        }
        originals = pd.DataFrame(
            [
                {"id": 0, "age": 30, "hours.per.week": 40, "education": " HS-grad"},
            ]
        )
        cfs = pd.DataFrame(
            [
                {
                    "id": 0,
                    "age": 31,
                    "hours.per.week": 40,
                    "education": " Bachelors",
                    "target_prob": 0.8,
                },
                {
                    "id": 0,
                    "age": 33,
                    "hours.per.week": 45,
                    "education": " HS-grad",
                    "target_prob": 0.9,
                },
                {
                    "id": 0,
                    "age": 29,
                    "hours.per.week": 40,
                    "education": " Bachelors",
                    "target_prob": 0.95,
                },
            ]
        )

        summary = build_evaluation_summary(cfs, originals, info, total_CFs=3)
        row = summary.iloc[0]

        self.assertEqual(row["valid_cf_count"], 3)
        self.assertEqual(row["feasible_cf_count"], 2)
        self.assertEqual(row["diversity_pair_count"], 3)
        self.assertAlmostEqual(row["diversity_cont"], 0.1)
        self.assertAlmostEqual(row["diversity_cat"], 2 / 3)
        self.assertAlmostEqual(row["diversity"], 13 / 45)
        pred_kernel = np.array(
            [
                [1.0 + 1e-6, 1.0 / 1.4, 1.0 / (1.0 + 1.0 / 30.0)],
                [1.0 / 1.4, 1.0 + 1e-6, 1.0 / (1.0 + 13.0 / 30.0)],
                [1.0 / (1.0 + 1.0 / 30.0), 1.0 / (1.0 + 13.0 / 30.0), 1.0 + 1e-6],
            ]
        )
        pred_dpp = np.linalg.det(pred_kernel)
        self.assertEqual(row["dpp_pair_count"], 3)
        self.assertAlmostEqual(row["dpp_diversity"], pred_dpp)
        self.assertAlmostEqual(row["log_dpp_diversity"], np.log(pred_dpp))
        feasible_kernel = np.array(
            [
                [1.0 + 1e-6, 1.0 / 1.4],
                [1.0 / 1.4, 1.0 + 1e-6],
            ]
        )
        feasible_dpp = np.linalg.det(feasible_kernel)
        self.assertEqual(row["feasible_diversity_pair_count"], 1)
        self.assertAlmostEqual(row["feasible_diversity_cont"], 0.1)
        self.assertAlmostEqual(row["feasible_diversity_cat"], 1.0)
        self.assertAlmostEqual(row["feasible_diversity"], 0.4)
        self.assertEqual(row["feasible_dpp_pair_count"], 1)
        self.assertAlmostEqual(row["feasible_dpp_diversity"], feasible_dpp)
        self.assertAlmostEqual(row["feasible_log_dpp_diversity"], np.log(feasible_dpp))
        self.assertEqual(row["raw_diversity_pair_count"], 3)
        self.assertAlmostEqual(row["raw_diversity_cont"], 0.1)
        self.assertAlmostEqual(row["raw_diversity_cat"], 2 / 3)
        self.assertAlmostEqual(row["raw_diversity"], 13 / 45)
        raw_kernel = np.array(
            [
                [1.0 + 1e-6, 1.0 / 1.4, 1.0 / (1.0 + 1.0 / 30.0)],
                [1.0 / 1.4, 1.0 + 1e-6, 1.0 / (1.0 + 13.0 / 30.0)],
                [1.0 / (1.0 + 1.0 / 30.0), 1.0 / (1.0 + 13.0 / 30.0), 1.0 + 1e-6],
            ]
        )
        raw_dpp = np.linalg.det(raw_kernel)
        self.assertEqual(row["raw_dpp_pair_count"], 3)
        self.assertAlmostEqual(row["raw_dpp_diversity"], raw_dpp)
        self.assertAlmostEqual(row["raw_log_dpp_diversity"], np.log(raw_dpp))

    def test_build_evaluation_summary_reports_latent_nn_for_valid_and_raw_cfs(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "target_col": "income",
            "column_names": ["age", "education", "income"],
            "integer": ["age"],
            "immutable": [],
            "monotone_increase": ["age"],
            "monotone_decrease": [],
            "column_domain": {
                "age": [20.0, 40.0],
                "education": [" HS-grad", " Bachelors"],
            },
        }
        originals = pd.DataFrame(
            [{"id": 0, "age": 30, "education": " HS-grad"}]
        )
        cfs = pd.DataFrame(
            [
                {
                    "id": 0,
                    "age": 31,
                    "education": " Bachelors",
                    "target_prob": 0.8,
                },
                {
                    "id": 0,
                    "age": 29,
                    "education": " Bachelors",
                    "target_prob": 0.9,
                },
            ]
        )
        cf_latents = np.array([[1.0, 1.0], [3.0, 0.0]])
        reference_latents = np.array([[0.0, 0.0], [5.0, 0.0]])

        summary = build_evaluation_summary(
            cfs,
            originals,
            info,
            total_CFs=2,
            cf_latents=cf_latents,
            reference_latents=reference_latents,
        )
        row = summary.iloc[0]

        self.assertEqual(row["valid_cf_count"], 2)
        self.assertEqual(row["latent_nn_count"], 2)
        self.assertAlmostEqual(row["latent_nn_distance"], (np.sqrt(2.0) + 2.0) / 2)
        self.assertEqual(row["feasible_cf_count"], 1)
        self.assertEqual(row["feasible_latent_nn_count"], 1)
        self.assertAlmostEqual(row["feasible_latent_nn_distance"], np.sqrt(2.0))
        self.assertEqual(row["raw_latent_nn_count"], 2)
        self.assertAlmostEqual(
            row["raw_latent_nn_distance"], (np.sqrt(2.0) + 2.0) / 2
        )

    def test_build_feature_change_summary_is_available_for_all_methods(self):
        info = {
            "num_col_idx": [0],
            "cat_col_idx": [1],
            "idx_name_mapping": {"0": "age", "1": "education"},
            "target_col": "income",
            "column_names": ["age", "education", "income"],
        }
        originals = pd.DataFrame(
            [
                {"id": 0, "age": 30, "education": " HS-grad"},
                {"id": 1, "age": 40, "education": " Bachelors"},
            ]
        )
        cfs = pd.DataFrame(
            [
                {"id": 0, "age": 31, "education": " Bachelors", "target_prob": 0.8},
                {"id": 1, "age": 40, "education": " Bachelors", "target_prob": 0.9},
            ]
        )

        summary = build_feature_change_summary(cfs, originals, info)
        by_feature = summary.set_index("feature")

        self.assertEqual(by_feature.loc["age", "feature_type"], "numerical")
        self.assertEqual(by_feature.loc["age", "changed_count"], 1)
        self.assertEqual(by_feature.loc["education", "feature_type"], "categorical")
        self.assertEqual(by_feature.loc["education", "top_transition"], " HS-grad ->  Bachelors")


if __name__ == "__main__":
    unittest.main()
