import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mdpcf.black_box_preprocessing import (
    BlackBoxPreprocessor,
    PROTOCOL,
    assert_encoded_input_alignment,
    guard_preprocessing_output,
    save_source_manifest,
)


@pytest.fixture
def data_dir(tmp_path):
    train = pd.DataFrame({"amount": [0.0, 10.0, 5.0, 4.0],
                          "status": [-2, -1, 2, 10], "label": [0, 1, 0, 1]})
    test = pd.DataFrame({"amount": [100.0, 5.0], "status": [-2, 10], "label": [1, 0]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    np.save(tmp_path / "X_num_train.npy", train[["amount"]].to_numpy(np.float32))
    np.save(tmp_path / "X_cat_train.npy", train[["status"]].astype(str).to_numpy(object))
    info = {"num_col_idx": [0], "cat_col_idx": [1], "target_col": "label",
            "idx_name_mapping": {"0": "amount", "1": "status", "2": "label"}}
    (tmp_path / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return tmp_path


def test_training_only_scaling_and_training_string_category_order(data_dir):
    preprocessor = BlackBoxPreprocessor(data_dir)
    test = pd.read_csv(data_dir / "test.csv")
    actual = preprocessor.transform(test)
    # Numeric -2/-1 and 2/10 must not be sorted numerically in the one-hot blocks.
    np.testing.assert_array_equal(preprocessor.encoder.categories_[0], [-1, -2, 10, 2])
    np.testing.assert_allclose(actual, [[10.0, 0, 1, 0, 0], [0.5, 0, 0, 1, 0]])
    preprocessor.scaler.fit(test[["amount"]])
    preprocessor.encoder.fit(test[["status"]])
    np.testing.assert_array_equal(actual, preprocessor.transform(test))


def test_encoder_inverse_transform_preserves_raw_numeric_categories(data_dir):
    preprocessor = BlackBoxPreprocessor(data_dir)
    frame = pd.read_csv(data_dir / "train.csv")
    encoded = preprocessor.encoder.transform(frame[["status"]])
    decoded = preprocessor.encoder.inverse_transform(encoded)
    np.testing.assert_array_equal(decoded[:, 0], frame.status.to_numpy())
    assert len(preprocessor.encoder.get_feature_names(["status"])) == 4


def test_unknown_raw_category_is_all_zero(data_dir):
    preprocessor = BlackBoxPreprocessor(data_dir)
    frame = pd.DataFrame({"amount": [2.0], "status": [99]})
    np.testing.assert_array_equal(preprocessor.transform(frame)[:, 1:], np.zeros((1, 4)))


def test_training_csv_misalignment_is_rejected(data_dir):
    train = pd.read_csv(data_dir / "train.csv").iloc[::-1]
    train.to_csv(data_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="not aligned"):
        BlackBoxPreprocessor(data_dir)


def test_alignment_check_rejects_permuted_features():
    matrix = np.array([[0, 1], [0.5, 0.2]])
    assert_encoded_input_alignment(matrix, matrix.copy())
    with pytest.raises(ValueError, match="does not match"):
        assert_encoded_input_alignment(matrix[:, ::-1], matrix)


def test_legacy_outputs_cannot_be_overwritten(tmp_path):
    args = SimpleNamespace()
    (tmp_path / "cfs.csv").write_text("id,target_prob\n", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="new save_path"):
        guard_preprocessing_output(args, tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"black_box_preprocessing_protocol": PROTOCOL}), encoding="utf-8")
    guard_preprocessing_output(args, tmp_path)
    assert args.black_box_preprocessing_protocol == PROTOCOL


def test_partial_legacy_run_cannot_reuse_old_model_cache(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="new save_path"):
        guard_preprocessing_output(SimpleNamespace(), tmp_path)


def test_saved_raw_features_preserve_probabilities_after_csv_roundtrip(data_dir):
    preprocessor = BlackBoxPreprocessor(data_dir)
    frame = pd.read_csv(data_dir / "test.csv")
    matrix = preprocessor.transform(frame)
    weights = np.array([0.01, -0.4, 0.9, 0.1, 0.2])
    frame["target_prob"] = 1 / (1 + np.exp(-(matrix @ weights)))
    path = data_dir / "cfs.csv"
    frame.to_csv(path, index=False)
    saved = pd.read_csv(path)
    recomputed = 1 / (1 + np.exp(-(preprocessor.transform(saved) @ weights)))
    np.testing.assert_allclose(recomputed, saved.target_prob, rtol=1e-7, atol=1e-9)


def test_source_manifest_records_global_test_indices(tmp_path):
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
    save_source_manifest(tmp_path, np.array([1, 2]), probs, 1)
    manifest = pd.read_csv(tmp_path / "source_manifest.csv")
    assert manifest.test_index.tolist() == [1, 2]
    assert manifest.black_box_pred.tolist() == [0, 0]
    np.testing.assert_allclose(manifest.target_prob, [0.2, 0.4])


