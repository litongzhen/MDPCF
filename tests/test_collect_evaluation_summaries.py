import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.collect_evaluation_summaries import collect_evaluation_summaries


class CollectEvaluationSummariesTest(unittest.TestCase):
    def test_collects_recursive_evaluation_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "adult" / "MDPCF" / "run1"
            second = root / "adult" / "DiCE" / "run2"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            pd.DataFrame([{"dataset": "adult", "method": "mdpcf", "coverage": 0.8}]).to_csv(
                first / "evaluation_summary.csv", index=False
            )
            pd.DataFrame([{"dataset": "adult", "method": "dice", "coverage": 0.6}]).to_csv(
                second / "evaluation_summary.csv", index=False
            )

            collected = collect_evaluation_summaries(root)

        self.assertEqual(len(collected), 2)
        self.assertIn("result_dir", collected.columns)
        self.assertEqual(sorted(collected["method"].tolist()), ["dice", "mdpcf"])


if __name__ == "__main__":
    unittest.main()
