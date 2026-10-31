"""Reuse the classifier's training representation across baseline adapters."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PROTOCOL = "train-fitted-v1"


class FrozenNumericalScaler:
    def __init__(self, columns, training):
        self.columns = list(columns)
        self.dtype = training.dtype
        self.scaler = MinMaxScaler().fit(training)

    def fit(self, frame):
        if list(frame.columns) != self.columns:
            raise ValueError("Numerical feature order differs from classifier training")
        return self

    def transform(self, frame):
        return self.scaler.transform(np.asarray(frame, dtype=self.dtype))

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(np.asarray(values, dtype=self.dtype))


class FrozenCategoricalEncoder:
    def __init__(self, columns, raw_training, encoded_training):
        self.columns = list(columns)
        self.encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        ).fit(encoded_training)
        self.raw_to_training = []
        self.training_to_raw = []
        for index, name in enumerate(self.columns):
            pairs = pd.DataFrame({"raw": raw_training[name],
                                  "training": encoded_training[:, index]}).drop_duplicates()
            if pairs.raw.duplicated().any() or pairs.training.duplicated().any():
                raise ValueError(f"Ambiguous raw/training category mapping for {name}")
            self.raw_to_training.append(dict(zip(pairs.raw, pairs.training)))
            self.training_to_raw.append(dict(zip(pairs.training, pairs.raw)))
        self.categories_ = [np.asarray([self.training_to_raw[i][value] for value in categories])
                            for i, categories in enumerate(self.encoder.categories_)]

    def fit(self, frame):
        if list(frame.columns) != self.columns:
            raise ValueError("Categorical feature order differs from classifier training")
        return self

    def transform(self, frame):
        values = np.asarray(frame, dtype=object)
        mapped = np.empty(values.shape, dtype=object)
        for i, lookup in enumerate(self.raw_to_training):
            # Unknown raw values must remain unknown, not alias an existing string.
            mapped[:, i] = [lookup.get(value, None) for value in values[:, i]]
        return self.encoder.transform(mapped)

    def inverse_transform(self, values):
        decoded = self.encoder.inverse_transform(np.asarray(values))
        raw = np.empty(decoded.shape, dtype=object)
        for i, lookup in enumerate(self.training_to_raw):
            raw[:, i] = [lookup.get(value, None) for value in decoded[:, i]]
        return raw

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)

    def get_feature_names(self, input_features=None):
        return self.get_feature_names_out(input_features)


class BlackBoxPreprocessor:
    def __init__(self, data_dir):
        folder = Path(data_dir)
        info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        names = {int(k): v for k, v in info["idx_name_mapping"].items()}
        self.numerical = [names[i] for i in info["num_col_idx"]]
        self.categorical = [names[i] for i in info["cat_col_idx"]]
        self.target = info["target_col"]
        train = pd.read_csv(folder / "train.csv")
        self.scaler = None
        self.encoder = None
        if self.numerical:
            numerical = np.load(folder / "X_num_train.npy", allow_pickle=True)
            if not np.allclose(train[self.numerical].to_numpy(float), numerical, equal_nan=True):
                raise ValueError("Training CSV and classifier numerical arrays are not aligned")
            if not np.isfinite(numerical).all():
                raise ValueError("Missing numerical values require the classifier's fitted imputer")
            self.scaler = FrozenNumericalScaler(self.numerical, numerical)
        if self.categorical:
            categorical = np.load(folder / "X_cat_train.npy", allow_pickle=True)
            self.encoder = FrozenCategoricalEncoder(self.categorical, train, categorical)

    def transform(self, frame):
        parts = []
        if self.scaler is not None:
            parts.append(self.scaler.transform(frame[self.numerical]))
        if self.encoder is not None:
            parts.append(self.encoder.transform(frame[self.categorical]))
        if not parts:
            raise ValueError("Classifier has no input features")
        return np.concatenate(parts, axis=1).astype(np.float32)


def assert_encoded_input_alignment(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise ValueError("Baseline input does not match the frozen classifier preprocessing")


def guard_preprocessing_output(args, output_dir):
    folder = Path(output_dir)
    config_path = folder / "config.json"
    if (folder / "cfs.csv").exists() or config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if previous.get("black_box_preprocessing_protocol") != PROTOCOL:
            raise ValueError("Legacy baseline results exist; use a new save_path instead of overwriting them")
        generation = getattr(args, "carla_generation_protocol", None)
        if generation is not None and previous.get("carla_generation_protocol") != generation:
            raise ValueError("Different generation protocol exists; use a new save_path")
    args.black_box_preprocessing_protocol = PROTOCOL


def save_source_manifest(output_dir, indices, probabilities, target):
    pd.DataFrame({
        "selected_order": np.arange(len(indices)),
        "test_index": indices,
        "black_box_pred": probabilities[indices].argmax(axis=1),
        "target_prob": probabilities[indices, target],
    }).to_csv(Path(output_dir) / "source_manifest.csv", index=False)
