import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def load_carla_source():
    repo_root = Path(__file__).resolve().parents[1]
    carla_root = repo_root / "baselines" / "CARLA"
    dice_root = repo_root / "baselines" / "dice" / "DiCE-official"
    if not carla_root.exists():
        raise FileNotFoundError(f"Local CARLA source tree not found: {carla_root}")
    if not dice_root.exists():
        raise FileNotFoundError(f"Local DiCE source tree not found: {dice_root}")

    for source_root in [dice_root, carla_root]:
        source_root_str = str(source_root)
        if source_root_str in sys.path:
            sys.path.remove(source_root_str)
        sys.path.insert(0, str(source_root))

    sys.modules.setdefault("tensorflow.contrib", types.ModuleType("tensorflow.contrib"))
    sys.modules.setdefault(
        "tensorflow.contrib.eager", types.ModuleType("tensorflow.contrib.eager")
    )
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(OneHotEncoder, "get_feature_names"):
        OneHotEncoder.get_feature_names = OneHotEncoder.get_feature_names_out
    original_one_hot_encoder = preprocessing.OneHotEncoder

    def compatible_one_hot_encoder(*args, **kwargs):
        if "sparse" in kwargs and "sparse_output" not in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        encoder = original_one_hot_encoder(*args, **kwargs)
        if not hasattr(encoder, "get_feature_names"):
            encoder.get_feature_names = encoder.get_feature_names_out
        return encoder

    preprocessing.OneHotEncoder = compatible_one_hot_encoder

    return carla_root


CARLA_ROOT = load_carla_source()

from carla.data.catalog.catalog import DataCatalog  # noqa: E402


class TrainTestCsvCatalog(DataCatalog):
    def __init__(
        self,
        train_path,
        test_path,
        continuous,
        categorical,
        immutables,
        target,
        scaling_method="MinMax",
        encoding_method="OneHot_drop_binary",
    ):
        self._continuous = continuous
        self._categorical = categorical
        self._immutables = immutables
        self._target = target

        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)
        raw = pd.concat([train_raw, test_raw], ignore_index=True)

        super().__init__(
            "custom", raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def continuous(self):
        return self._continuous

    @property
    def categorical(self):
        return self._categorical

    @property
    def immutables(self):
        return self._immutables

    @property
    def target(self):
        return self._target
