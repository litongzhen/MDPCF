import unittest

import numpy as np

from mdpcf.sample import build_sample_tag, select_source_indices


class SourceSelectionTest(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0])

    def test_source_start_selects_a_disjoint_slice_of_eligible_sources(self):
        formal = select_source_indices(
            self.predictions, target_class_index=1, num_samples=3, source_start=0
        )
        probe = select_source_indices(
            self.predictions, target_class_index=1, num_samples=2, source_start=3
        )

        np.testing.assert_array_equal(formal, np.array([0, 2, 3]))
        np.testing.assert_array_equal(probe, np.array([5, 6]))
        self.assertTrue(set(formal).isdisjoint(set(probe)))

    def test_source_selection_fails_when_requested_slice_is_incomplete(self):
        with self.assertRaisesRegex(ValueError, "Requested 3 source instances"):
            select_source_indices(
                self.predictions,
                target_class_index=1,
                num_samples=3,
                source_start=4,
            )

    def test_source_selection_rejects_negative_start(self):
        with self.assertRaisesRegex(ValueError, "source_start must be non-negative"):
            select_source_indices(
                self.predictions,
                target_class_index=1,
                num_samples=2,
                source_start=-1,
            )

    def test_probe_sample_tag_is_isolated_from_formal_output(self):
        self.assertEqual(build_sample_tag(20, 10, source_start=0), "20_samples_CFs_10")
        self.assertEqual(
            build_sample_tag(20, 10, source_start=100),
            "20_samples_from_100_CFs_10",
        )


if __name__ == "__main__":
    unittest.main()
