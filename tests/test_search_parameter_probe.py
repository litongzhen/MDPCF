import unittest

from scripts.summarize_csvae_search_parameter_probe import (
    rank_configurations,
    rank_fallback_configurations,
)


class SearchParameterProbeRankingTest(unittest.TestCase):
    def test_ranking_rejects_rows_below_hard_gates(self):
        rows = [
            self._row("eligible", coverage=1.0, validity_expected=0.9),
            self._row("low_coverage", coverage=0.9, validity_expected=1.0),
            self._row("low_yield", coverage=1.0, validity_expected=0.89),
        ]

        ranked = rank_configurations(rows)

        self.assertEqual([row["config_id"] for row in ranked], ["eligible"])

    def test_ranking_uses_target_diversity_and_low_lof_directions(self):
        rows = [
            self._row("balanced", target=0.8, diversity=0.4, lof=1.0),
            self._row("worse", target=0.7, diversity=0.3, lof=2.0),
        ]

        ranked = rank_configurations(rows)

        self.assertEqual([row["config_id"] for row in ranked], ["balanced", "worse"])
        self.assertLess(ranked[0]["mean_rank"], ranked[1]["mean_rank"])

    def test_proximity_breaks_equal_mean_rank(self):
        rows = [
            self._row("far", proximity=0.5),
            self._row("near", proximity=0.2),
        ]

        ranked = rank_configurations(rows)

        self.assertEqual([row["config_id"] for row in ranked], ["near", "far"])

    def test_fallback_ranking_includes_yield_without_lowering_final_gate(self):
        rows = [
            self._row("high_yield", validity_expected=0.8, target=0.7),
            self._row("low_yield", validity_expected=0.6, target=0.9),
        ]

        ranked = rank_fallback_configurations(rows)

        self.assertEqual(ranked[0]["config_id"], "high_yield")
        self.assertIn("yield_rank", ranked[0])

    @staticmethod
    def _row(
        config_id,
        coverage=1.0,
        validity_expected=1.0,
        target=0.8,
        diversity=0.4,
        lof=1.0,
        proximity=0.3,
    ):
        return {
            "dataset": "adult",
            "config_id": config_id,
            "coverage": coverage,
            "validity_expected": validity_expected,
            "mean_target_prob": target,
            "diversity": diversity,
            "lof_log_scale": lof,
            "proximity": proximity,
        }


if __name__ == "__main__":
    unittest.main()
