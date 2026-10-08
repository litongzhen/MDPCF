import json
from pathlib import Path

from scripts.prepare_release_summaries import sanitize_config
from scripts.export_environment import environment_snapshot


def test_config_export_excludes_machine_local_and_derived_paths():
    config = {"seed": 42, "save_path": "C:/private/results", "device": "cuda:0",
              "reference_config": "C:/private/config.json", "mode": "sample"}
    assert sanitize_config(config, {"seed", "save_path", "mode"}) == {"seed": 42, "mode": "sample"}


def test_environment_snapshot_records_versions_not_local_package_paths():
    snapshot = environment_snapshot()
    assert "numpy" in snapshot["packages"]
    serialized = json.dumps(snapshot)
    assert "file:///" not in serialized
    assert "Users/" not in serialized


def test_default_requirements_do_not_pull_optional_baselines():
    root = Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "torch==2.10.0" in lines
    assert not any(line.startswith(("tensorflow", "neptune", "nflows")) for line in lines)
