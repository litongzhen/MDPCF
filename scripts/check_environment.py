"""Check imports without loading data or starting experiments."""

import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--skip-dicoflex", action="store_true",
                        help="Check redistributed baselines without separately obtained DiCoFlex")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The reference environment uses Python 3.10")
    modules = ["utils", "utils_train", "mdpcf.csvae.model", "mdpcf.sample",
               "evaluation_framework.evaluate"]
    if args.baselines:
        modules += ["baselines.dice.sample", "baselines.cchvae.sample",
                    "baselines.revise.sample", "baselines.wachter.sample"]
        if not args.skip_dicoflex:
            modules.append("baselines.dicoflex.sample")
    for name in modules:
        importlib.import_module(name)
        print(f"OK {name}")
    import torch
    print(f"Python {sys.version.split()[0]}; Torch {torch.__version__}; CUDA {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
