import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mdpcf.checkpoints import (
    aggregate_sampling_stats,
    is_resumable_sampling_directory,
    load_completed_sources,
    merge_source_shards,
    write_source_checkpoint,
)


class SamplingCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_root = Path(self.temp_dir.name) / "checkpoints"
        self.stats = {
            "attempted_count": 1,
            "optimized_valid_source_count": 1,
            "optimized_valid_candidate_count": 2,
            "post_projection_candidate_count": 2,
            "post_projection_validity_count": 2,
            "post_projection_constraints_count": 2,
            "saved_cf_count": 2,
            "saved_source_count": 1,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_completed_marker_is_written_only_after_source_shards(self):
        cfs = pd.DataFrame({"id": [2], "age": [35], "target_prob": [0.8]})
        latents = pd.DataFrame({"id": [2], "z_0": [0.1]})

        write_source_checkpoint(
            self.checkpoint_root,
            source_id=2,
            stats=self.stats,
            shards={"cfs": cfs, "latents": latents},
        )

        completed = load_completed_sources(self.checkpoint_root)

        self.assertEqual(set(completed), {2})
        self.assertEqual(completed[2]["stats"], self.stats)
        self.assertTrue((self.checkpoint_root / "cfs" / "2.csv").is_file())
        self.assertTrue((self.checkpoint_root / "latents" / "2.csv").is_file())
        self.assertFalse(list(self.checkpoint_root.rglob("*.tmp")))

    def test_checkpoint_directory_marks_an_existing_result_path_as_resumable(self):
        result_dir = Path(self.temp_dir.name) / "result"
        result_dir.mkdir()
        (result_dir / "cfs.csv").write_text("id\n", encoding="utf-8")

        self.assertFalse(is_resumable_sampling_directory(result_dir))

        (result_dir / "checkpoints").mkdir()

        self.assertTrue(is_resumable_sampling_directory(result_dir))

    def test_malformed_or_inconsistent_marker_is_not_resumed(self):
        marker_dir = self.checkpoint_root / "sources"
        marker_dir.mkdir(parents=True)
        (marker_dir / "3.json").write_text("not json", encoding="utf-8")
        (marker_dir / "4.json").write_text(
            json.dumps({"source_id": 999, "completed": True, "stats": {}}),
            encoding="utf-8",
        )

        self.assertEqual(load_completed_sources(self.checkpoint_root), {})

    def test_marker_with_a_missing_declared_shard_is_not_resumed(self):
        marker_dir = self.checkpoint_root / "sources"
        marker_dir.mkdir(parents=True)
        (marker_dir / "5.json").write_text(
            json.dumps(
                {
                    "source_id": 5,
                    "completed": True,
                    "stats": self.stats,
                    "shards": ["cfs"],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(load_completed_sources(self.checkpoint_root), {})

    def test_merge_orders_shards_by_source_id_and_uses_completed_markers(self):
        write_source_checkpoint(
            self.checkpoint_root,
            source_id=10,
            stats=self.stats,
            shards={"cfs": pd.DataFrame({"id": [10], "value": ["ten"]})},
        )
        write_source_checkpoint(
            self.checkpoint_root,
            source_id=2,
            stats=self.stats,
            shards={"cfs": pd.DataFrame({"id": [2], "value": ["two"]})},
        )
        (self.checkpoint_root / "cfs" / "99.csv").write_text(
            "id,value\n99,ignored\n", encoding="utf-8"
        )

        merged = merge_source_shards(
            self.checkpoint_root,
            "cfs",
            columns=["id", "value"],
        )

        self.assertEqual(merged["id"].tolist(), [2, 10])
        self.assertEqual(merged["value"].tolist(), ["two", "ten"])

    def test_merge_without_a_fixed_schema_preserves_shard_columns(self):
        write_source_checkpoint(
            self.checkpoint_root,
            source_id=1,
            stats=self.stats,
            shards={"projection": pd.DataFrame({"id": [1], "shift": [0.2]})},
        )

        merged = merge_source_shards(self.checkpoint_root, "projection", columns=None)

        self.assertEqual(merged.columns.tolist(), ["id", "shift"])
        self.assertEqual(merged.to_dict(orient="records"), [{"id": 1, "shift": 0.2}])

    def test_aggregate_sampling_stats_includes_completed_empty_sources(self):
        empty_stats = dict(self.stats)
        empty_stats.update(
            {
                "optimized_valid_source_count": 0,
                "optimized_valid_candidate_count": 0,
                "post_projection_candidate_count": 0,
                "post_projection_validity_count": 0,
                "post_projection_constraints_count": 0,
                "saved_cf_count": 0,
                "saved_source_count": 0,
            }
        )
        write_source_checkpoint(
            self.checkpoint_root,
            source_id=0,
            stats=empty_stats,
            shards={},
        )
        write_source_checkpoint(
            self.checkpoint_root,
            source_id=1,
            stats=self.stats,
            shards={},
        )

        totals = aggregate_sampling_stats(load_completed_sources(self.checkpoint_root))

        self.assertEqual(totals["attempted_count"], 2)
        self.assertEqual(totals["saved_source_count"], 1)
        self.assertEqual(totals["saved_cf_count"], 2)


if __name__ == "__main__":
    unittest.main()
