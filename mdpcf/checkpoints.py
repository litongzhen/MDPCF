import json
import os
import uuid
from pathlib import Path

import pandas as pd


def is_resumable_sampling_directory(save_path):
    return (Path(save_path) / "checkpoints").is_dir()


def _checkpoint_path(checkpoint_root, shard_name, source_id, suffix):
    return Path(checkpoint_root) / shard_name / f"{int(source_id)}{suffix}"


def _atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_write_dataframe(frame, path):
    temporary_path = _atomic_path(path)
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _json_compatible(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _atomic_write_json(payload, path):
    temporary_path = _atomic_path(path)
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(temporary_path, path)


def write_source_checkpoint(checkpoint_root, source_id, stats, shards):
    """Atomically commit all result shards before marking one source complete."""
    source_id = int(source_id)
    committed_shards = []
    for shard_name, frame in shards.items():
        if frame is None:
            continue
        shard_path = _checkpoint_path(checkpoint_root, shard_name, source_id, ".csv")
        _atomic_write_dataframe(frame, shard_path)
        committed_shards.append(shard_name)

    marker = {
        "source_id": source_id,
        "completed": True,
        "stats": {key: _json_compatible(value) for key, value in stats.items()},
        "shards": sorted(committed_shards),
    }
    marker_path = _checkpoint_path(checkpoint_root, "sources", source_id, ".json")
    _atomic_write_json(marker, marker_path)


def load_completed_sources(checkpoint_root):
    """Return valid source completion markers keyed by their integer source id."""
    marker_dir = Path(checkpoint_root) / "sources"
    if not marker_dir.is_dir():
        return {}

    completed = {}
    for marker_path in marker_dir.glob("*.json"):
        try:
            source_id = int(marker_path.stem)
            with marker_path.open(encoding="utf-8") as handle:
                marker = json.load(handle)
            if (
                not isinstance(marker, dict)
                or marker.get("completed") is not True
                or marker.get("source_id") != source_id
                or not isinstance(marker.get("stats"), dict)
                or not isinstance(marker.get("shards", []), list)
            ):
                continue
            if any(
                not isinstance(shard_name, str)
                or not _checkpoint_path(
                    checkpoint_root, shard_name, source_id, ".csv"
                ).is_file()
                for shard_name in marker.get("shards", [])
            ):
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        completed[source_id] = marker
    return completed


def merge_source_shards(checkpoint_root, shard_name, columns):
    """Merge shards for completed sources in source-id order."""
    completed = load_completed_sources(checkpoint_root)
    frames = []
    for source_id in sorted(completed):
        shard_path = _checkpoint_path(checkpoint_root, shard_name, source_id, ".csv")
        if shard_path.is_file():
            frames.append(pd.read_csv(shard_path))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def aggregate_sampling_stats(completed_sources):
    """Sum per-source counters from validated completion markers."""
    totals = {}
    for marker in completed_sources.values():
        for key, value in marker["stats"].items():
            totals[key] = totals.get(key, 0) + value
    return totals
