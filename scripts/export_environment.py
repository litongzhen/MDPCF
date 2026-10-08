"""Save observed package versions without local paths or credentials."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform


def environment_snapshot():
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    snapshot = {"python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(), "packages": dict(sorted(packages.items())),
                "purpose": "Observed environment, not a portable dependency lock"}
    try:
        import torch
        snapshot["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                             "cuda_available": torch.cuda.is_available()}
    except ImportError:
        snapshot["torch"] = None
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(environment_snapshot(), indent=2), encoding="utf-8")
    print(f"Saved environment snapshot: {args.output}")


if __name__ == "__main__":
    main()
