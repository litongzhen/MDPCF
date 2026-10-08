import joblib
import os
import sys
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils_train import (
    get_name_form_args,
    get_optim_params_form_args,
)
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler


from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.single_column import TVComplement
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
from joblib import dump, load
from pyod.models.dif import DIF


# print(clf.score_samples([[0.1], [0], [90]]))


def get_outlier_detection_model(
    train_df, data_dir, info, target_only=False, negative_only=False
):

    if target_only:
        model_filename = f"{data_dir}/outlier_model_eval_target_only.pkl"
    elif negative_only:
        model_filename = f"{data_dir}/outlier_model_eval_negative_only.pkl"
    else:
        model_filename = f"{data_dir}/outlier_model_eval.pkl"

    if os.path.isfile(model_filename):
        with open(model_filename, "rb") as file:
            clf_IF = pickle.load(file)
    else:
        num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
        cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

        numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        transformations = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        clf_IF = Pipeline(
            steps=[
                ("preprocessor", transformations),
                ("classifier", IsolationForest(random_state=42)),
            ]
        )

        clf_IF.fit(train_df)

        with open(model_filename, "wb") as file:
            pickle.dump(clf_IF, file)

    return clf_IF


def get_outlier_detection_model_pyod(
    train_df, data_dir, info, target_only=False, negative_only=False
):

    if target_only:
        model_filename = f"{data_dir}/outlier_model_eval_target_only_pyod.joblib"
    elif negative_only:
        model_filename = f"{data_dir}/outlier_model_eval_negative_only_pyod.pkl"
    else:
        model_filename = f"{data_dir}/outlier_model_eval_pyod.joblib"

    if os.path.isfile(model_filename):
        clf_IF = load(model_filename)
    else:
        num_cols = [info["column_names"][x] for x in info["num_col_idx"]]
        cat_cols = [info["column_names"][x] for x in info["cat_col_idx"]]

        numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        transformations = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            sparse_threshold=0,
        )

        clf_IF = Pipeline(
            steps=[
                ("preprocessor", transformations),
                ("classifier", DIF(random_state=42)),
            ]
        )

        clf_IF.fit(train_df)

        dump(clf_IF, model_filename)

    return clf_IF


def get_plausibility_stats(
    correct_cfs,
    train_set_path,
    info,
    data_dir,
    target_only=False,
    pyod=False,
    negative_only=False,
):

    metadata = info["metadata"].copy()
    metadata["columns"] = {
        info["column_names"][int(key)]: value
        for key, value in metadata["columns"].items()
    }

    train_df = pd.read_csv(train_set_path)

    if target_only:
        train_df = train_df[train_df[info["target_col"]] == info["target_class"]]

    if negative_only:
        train_df = train_df[train_df[info["target_col"]] == info["negative_class"]]

    train_df = train_df.drop(info["target_col"], axis=1)

    if pyod:
        clf = get_outlier_detection_model_pyod(
            train_df,
            data_dir,
            info,
            target_only=target_only,
            negative_only=negative_only,
        )
    else:
        clf = get_outlier_detection_model(
            train_df,
            data_dir,
            info,
            target_only=target_only,
            negative_only=negative_only,
        )

    metadata["columns"].pop(info["target_col"])

    qual_report = QualityReport()
    qual_report.generate(train_df, correct_cfs, metadata, verbose=False)

    # diag_report = DiagnosticReport()
    # diag_report.generate(train_df, correct_cfs, metadata)

    # shapes = qual_report.get_details(property_name='Column Shapes')

    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns

    # trends = qual_report.get_details(property_name='Column Pair Trends')
    # print(trends)

    quality = qual_report.get_properties()
    Shape = quality["Score"][
        0
    ]  # Kolmogorov-Smirnov statistic for discrete, TVComplement for discrete
    Trend = quality["Score"][
        1
    ]  # CorrelationSimilarity for continuous-continuous, ContingencySimilarity for discrete-continuous, discrete-discrete

    if pyod:
        anomaly_score = np.mean(clf.predict_proba(correct_cfs, method="unify")[:, 0])
    else:
        anomaly_score = np.mean(
            clf.decision_function(correct_cfs)
        )  # The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.

    return (Shape, Trend, anomaly_score)


def gower_distance(x1, x2, cat_col_idx):
    is_categorical = np.isin(np.arange(len(x1)), cat_col_idx)

    is_numeric = ~is_categorical

    # categorical columns
    sij_cat = np.where(
        x1[is_categorical] == x2[is_categorical],
        np.zeros_like(x1[is_categorical]),
        np.ones_like(x1[is_categorical]),
    )
    sum_cat = sij_cat.sum()

    # numerical columns
    abs_delta = np.absolute(x1[is_numeric] - x2[is_numeric])
    sij_num = abs_delta  # already normalized

    sum_num = sij_num.sum()
    sums = np.add(sum_cat, sum_num)
    feature_weight_sum = len(x1)
    sum_sij = np.divide(sums, feature_weight_sum)
    return sum_sij


def l0_distance(row1, row2):
    return (row1 != row2).sum()


def get_tabcf_original_name_form_args(args):
    opts_vae = [
        f"num_encoding[{args.num_encoding}]",
        f"num_loss[{args.num_reconstr_loss}]",
    ]

    if args.kl_weight > 0:
        opts_vae.append(f"kl_weight[{args.kl_weight}]")

    if args.gumbel_softmax:
        opts_vae.append(f"cat_actv[gumbel_softmax, tau={args.tau}]")

    opts_vae.append("cont_actv[sigmoid]" if args.sigmoid else "cont_actv[linear]")

    if getattr(args, "reparam", True):
        opts_vae.append("reparam")

    return "_".join(opts_vae)


def get_tabcf_original_optim_params_form_args(args):
    optimization_parameters = f"proximity_weight_input[{args.proximity_weight_input}]"

    if args.hidden_dims is not None:
        optimization_parameters = f"{optimization_parameters}_hidden_dims[{args.hidden_dims}]"

    if args.proximity_weight_latent > 0:
        optimization_parameters = (
            f"{optimization_parameters}_proximity_weight_latent"
            f"[{args.proximity_weight_latent}]_proximity_latent_loss"
            f"[{args.proximity_latent_loss}]"
        )

    if args.plausibility_weight_latent > 0:
        optimization_parameters = (
            f"{optimization_parameters}_plausibility_weight_latent"
            f"[{args.plausibility_weight_latent}]_plausibility_latent_loss"
            f"[{args.plausibility_latent_loss}]"
        )

    if args.proximity_input_loss is not None:
        optimization_parameters = (
            f"{optimization_parameters}_proximity_input_loss"
            f"[{args.proximity_input_loss}]"
        )

    if args.immutable:
        optimization_parameters = f"{optimization_parameters}_immutable[{args.immutable}]"

    return optimization_parameters


def _feature_names(info):
    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    if target_col in column_names:
        column_names.remove(target_col)
    return column_names


def _num_cat_cols(info):
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    num_cols = [idx_name_mapping[idx] for idx in info.get("num_col_idx", [])]
    cat_cols = [idx_name_mapping[idx] for idx in info.get("cat_col_idx", [])]
    return num_cols, cat_cols


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def compute_lof_log_scale_metrics(
    valid_cfs,
    originals_by_id,
    train_df,
    info,
    n_neighbors=20,
):
    feature_names = _feature_names(info)
    num_cols, cat_cols = _num_cat_cols(info)
    empty_metrics = {
        "lof_log_scale": np.nan,
        "lof_score_cf": np.nan,
        "lof_score_original": np.nan,
        "lof_count": 0,
    }

    if train_df is None or valid_cfs.empty:
        return empty_metrics

    train_features = train_df.copy()
    target_col = info["target_col"]
    if target_col in train_features.columns:
        train_features = train_features.drop(columns=[target_col])

    missing_cols = [col for col in feature_names if col not in train_features.columns]
    if missing_cols:
        return empty_metrics

    original_rows = []
    for _, cf_row in valid_cfs.iterrows():
        original_row = originals_by_id.get(cf_row["id"])
        if original_row is None:
            return empty_metrics
        original_rows.append({feature: original_row[feature] for feature in feature_names})

    if len(train_features) < 2 or not original_rows:
        return empty_metrics

    transformers = []
    if num_cols:
        transformers.append(("num", MinMaxScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", _make_one_hot_encoder(), cat_cols))

    if not transformers:
        return empty_metrics

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0)
    X_train = preprocessor.fit_transform(train_features[feature_names])
    X_cf = preprocessor.transform(valid_cfs[feature_names])
    X_original = preprocessor.transform(pd.DataFrame(original_rows))

    effective_neighbors = min(n_neighbors, len(train_features) - 1)
    if effective_neighbors < 1:
        return empty_metrics

    lof = LocalOutlierFactor(n_neighbors=effective_neighbors, novelty=True)
    lof.fit(X_train)
    cf_scores = -lof.score_samples(X_cf)
    original_scores = -lof.score_samples(X_original)
    cf_score = float(np.mean(cf_scores))

    return {
        "lof_log_scale": cf_score,
        "lof_score_cf": cf_score,
        "lof_score_original": float(np.mean(original_scores)),
        "lof_count": int(len(cf_scores)),
    }


def validate_constraints_for_evaluation(cfs, originals_by_id, info):
    valid = np.ones(len(cfs), dtype=bool)

    for row_pos, (_, cf_row) in enumerate(cfs.iterrows()):
        original_row = originals_by_id.get(cf_row["id"])
        if original_row is None:
            valid[row_pos] = False
            continue

        for col_name in info.get("immutable", []):
            if col_name in cfs.columns and cf_row[col_name] != original_row[col_name]:
                valid[row_pos] = False

        for col_name in info.get("monotone_increase", []):
            if col_name in cfs.columns and float(cf_row[col_name]) < float(original_row[col_name]):
                valid[row_pos] = False

        for col_name in info.get("monotone_decrease", []):
            if col_name in cfs.columns and float(cf_row[col_name]) > float(original_row[col_name]):
                valid[row_pos] = False

        for col_name in info.get("integer", []):
            if col_name in cfs.columns and not np.isclose(float(cf_row[col_name]), round(float(cf_row[col_name]))):
                valid[row_pos] = False

        for col_name, domain in info.get("column_domain", {}).items():
            if col_name not in cfs.columns:
                continue
            if isinstance(domain, list) and len(domain) == 2 and all(
                isinstance(value, (int, float)) for value in domain
            ):
                value = float(cf_row[col_name])
                if value < float(domain[0]) or value > float(domain[1]):
                    valid[row_pos] = False
            elif cf_row[col_name] not in domain:
                valid[row_pos] = False

        ordinal = info.get("ordinal", {})
        for col_name in info.get("ordinal_monotone_increase", []):
            if col_name not in cfs.columns or col_name not in ordinal:
                continue
            ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
            if ranks.get(cf_row[col_name], -1) < ranks.get(original_row[col_name], 0):
                valid[row_pos] = False

        for col_name in info.get("ordinal_monotone_decrease", []):
            if col_name not in cfs.columns or col_name not in ordinal:
                continue
            ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
            if ranks.get(cf_row[col_name], 10**9) > ranks.get(original_row[col_name], 0):
                valid[row_pos] = False

    return valid


def _normalized_numeric_distance(row_a, row_b, feature, info):
    num_min = float(info.get("num_min", {}).get(feature, 0.0))
    num_max = float(info.get("num_max", {}).get(feature, 1.0))
    denom = max(num_max - num_min, 1e-12)
    return abs(float(row_a[feature]) - float(row_b[feature])) / denom


def _mixed_distance(row_a, row_b, feature_names, num_cols, cat_cols, info):
    cont_distances = []
    cat_distances = []

    for feature in feature_names:
        if feature in num_cols:
            cont_distances.append(
                _normalized_numeric_distance(row_a, row_b, feature, info)
            )
        elif feature in cat_cols:
            cat_distances.append(float(row_a[feature] != row_b[feature]))

    all_distances = cont_distances + cat_distances
    return {
        "diversity": float(np.mean(all_distances)) if all_distances else np.nan,
        "diversity_cont": float(np.mean(cont_distances)) if cont_distances else np.nan,
        "diversity_cat": float(np.mean(cat_distances)) if cat_distances else np.nan,
    }


def compute_diversity_metrics(valid_cfs, feature_names, num_cols, cat_cols, info):
    source_diversities = []
    source_cont_diversities = []
    source_cat_diversities = []
    pair_count = 0

    for _, group in valid_cfs.groupby("id", sort=False):
        if len(group) < 2:
            continue

        pair_distances = []
        pair_cont_distances = []
        pair_cat_distances = []
        rows = list(group.iterrows())

        for left_idx in range(len(rows)):
            row_a = rows[left_idx][1]
            for right_idx in range(left_idx + 1, len(rows)):
                row_b = rows[right_idx][1]
                distances = _mixed_distance(
                    row_a, row_b, feature_names, num_cols, cat_cols, info
                )
                pair_distances.append(distances["diversity"])
                pair_cont_distances.append(distances["diversity_cont"])
                pair_cat_distances.append(distances["diversity_cat"])
                pair_count += 1

        source_diversities.append(float(np.nanmean(pair_distances)))
        source_cont_diversities.append(float(np.nanmean(pair_cont_distances)))
        source_cat_diversities.append(float(np.nanmean(pair_cat_distances)))

    return {
        "diversity": (
            float(np.nanmean(source_diversities)) if source_diversities else np.nan
        ),
        "diversity_cont": (
            float(np.nanmean(source_cont_diversities))
            if source_cont_diversities
            else np.nan
        ),
        "diversity_cat": (
            float(np.nanmean(source_cat_diversities))
            if source_cat_diversities
            else np.nan
        ),
        "diversity_pair_count": pair_count,
    }


def compute_dpp_diversity_metrics(valid_cfs, feature_names, num_cols, cat_cols, info, eps=1e-6):
    source_dpps = []
    source_log_dpps = []
    pair_count = 0

    for _, group in valid_cfs.groupby("id", sort=False):
        if len(group) < 2:
            continue

        rows = list(group.iterrows())
        kernel = np.eye(len(rows), dtype=np.float64) * (1.0 + eps)

        for left_idx in range(len(rows)):
            row_a = rows[left_idx][1]
            for right_idx in range(left_idx + 1, len(rows)):
                row_b = rows[right_idx][1]
                distance = _mixed_distance(
                    row_a, row_b, feature_names, num_cols, cat_cols, info
                )["diversity"]
                similarity = 1.0 / (1.0 + distance)
                kernel[left_idx, right_idx] = similarity
                kernel[right_idx, left_idx] = similarity
                pair_count += 1

        sign, log_dpp = np.linalg.slogdet(kernel)
        if sign <= 0:
            dpp = np.nan
            log_dpp = np.nan
        else:
            dpp = float(np.exp(log_dpp))
            log_dpp = float(log_dpp)
        source_dpps.append(dpp)
        source_log_dpps.append(log_dpp)

    return {
        "dpp_diversity": float(np.nanmean(source_dpps)) if source_dpps else np.nan,
        "log_dpp_diversity": (
            float(np.nanmean(source_log_dpps)) if source_log_dpps else np.nan
        ),
        "dpp_pair_count": pair_count,
    }


def compute_nearest_neighbor_distance(query_latents, reference_latents):
    query = np.asarray(query_latents, dtype=np.float64)
    reference = np.asarray(reference_latents, dtype=np.float64)

    if query.size == 0 or reference.size == 0:
        return {"latent_nn_distance": np.nan, "latent_nn_count": 0}

    query = query.reshape(query.shape[0], -1)
    reference = reference.reshape(reference.shape[0], -1)

    if query.shape[1] != reference.shape[1]:
        return {"latent_nn_distance": np.nan, "latent_nn_count": 0}

    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    return {
        "latent_nn_distance": float(nearest.mean()),
        "latent_nn_count": int(len(nearest)),
    }


def _empty_latent_metrics():
    return {"latent_nn_distance": np.nan, "latent_nn_count": 0}


def _compute_row_level_change_metrics(cfs_subset, originals_by_id, feature_names, num_cols, cat_cols, info):
    rows = []
    for _, cf_row in cfs_subset.iterrows():
        original_row = originals_by_id[cf_row["id"]]
        changed_all = []
        changed_num = []
        changed_cat = []
        prox_num_values = []

        for feature in feature_names:
            changed = cf_row[feature] != original_row[feature]
            changed_all.append(changed)
            if feature in num_cols:
                num_min = float(info.get("num_min", {}).get(feature, 0.0))
                num_max = float(info.get("num_max", {}).get(feature, 1.0))
                denom = max(num_max - num_min, 1e-12)
                prox_num_values.append(
                    abs(float(cf_row[feature]) - float(original_row[feature])) / denom
                )
                changed_num.append(changed)
            elif feature in cat_cols:
                changed_cat.append(changed)

        prox_num = float(np.mean(prox_num_values)) if prox_num_values else np.nan
        prox_cat = float(np.mean(changed_cat)) if changed_cat else np.nan
        proximity_values = [
            *(prox_num_values if prox_num_values else []),
            *([float(value) for value in changed_cat] if changed_cat else []),
        ]
        rows.append(
            {
                "sparsity": float(np.mean(changed_all)) if changed_all else np.nan,
                "sparsity_cont": float(np.mean(changed_num)) if changed_num else np.nan,
                "sparsity_cat": float(np.mean(changed_cat)) if changed_cat else np.nan,
                "proximity": float(np.mean(proximity_values)) if proximity_values else np.nan,
                "proximity_cont": prox_num,
                "proximity_cat": prox_cat,
            }
        )

    return pd.DataFrame(rows)


def _mean_metric(metrics_df, name, count):
    return float(metrics_df[name].mean()) if count else np.nan


def build_evaluation_summary(
    cfs,
    original_test_samples,
    info,
    total_CFs=1,
    threshold=0.5,
    cf_latents=None,
    reference_latents=None,
    train_df=None,
    enforce_constraints=True,
):
    feature_names = _feature_names(info)
    num_cols, cat_cols = _num_cat_cols(info)
    originals_by_id = {
        row["id"]: row for _, row in original_test_samples.iterrows()
    }

    attempted_count = len(original_test_samples)
    generated_cf_count = len(cfs)
    expected_cf_count = attempted_count * total_CFs

    if generated_cf_count == 0:
        return pd.DataFrame(
            [
                {
                    "attempted_count": attempted_count,
                    "generated_cf_count": 0,
                    "expected_cf_count": expected_cf_count,
                    "valid_cf_count": 0,
                    "valid_source_count": 0,
                    "coverage": 0.0,
                    "validity": 0.0,
                    "validity_actual": 0.0,
                    "validity_generated": 0.0,
                    "validity_expected": 0.0,
                    "mean_target_prob": np.nan,
                    "sparsity": np.nan,
                    "sparsity_cont": np.nan,
                    "sparsity_cat": np.nan,
                    "proximity": np.nan,
                    "proximity_cont": np.nan,
                    "proximity_cat": np.nan,
                    "diversity": np.nan,
                    "diversity_cont": np.nan,
                    "diversity_cat": np.nan,
                    "diversity_pair_count": 0,
                    "dpp_diversity": np.nan,
                    "log_dpp_diversity": np.nan,
                    "dpp_pair_count": 0,
                    "raw_diversity": np.nan,
                    "raw_diversity_cont": np.nan,
                    "raw_diversity_cat": np.nan,
                    "raw_diversity_pair_count": 0,
                    "raw_dpp_diversity": np.nan,
                    "raw_log_dpp_diversity": np.nan,
                    "raw_dpp_pair_count": 0,
                    "latent_nn_distance": np.nan,
                    "latent_nn_count": 0,
                    "raw_latent_nn_distance": np.nan,
                    "raw_latent_nn_count": 0,
                    "feasible_cf_count": 0,
                    "feasible_source_count": 0,
                    "feasible_coverage": 0.0,
                    "feasible_validity": 0.0,
                    "feasible_validity_actual": 0.0,
                    "feasible_validity_generated": 0.0,
                    "feasible_validity_expected": 0.0,
                    "feasible_mean_target_prob": np.nan,
                    "feasible_sparsity": np.nan,
                    "feasible_sparsity_cont": np.nan,
                    "feasible_sparsity_cat": np.nan,
                    "feasible_proximity": np.nan,
                    "feasible_proximity_cont": np.nan,
                    "feasible_proximity_cat": np.nan,
                    "feasible_diversity": np.nan,
                    "feasible_diversity_cont": np.nan,
                    "feasible_diversity_cat": np.nan,
                    "feasible_diversity_pair_count": 0,
                    "feasible_dpp_diversity": np.nan,
                    "feasible_log_dpp_diversity": np.nan,
                    "feasible_dpp_pair_count": 0,
                    "feasible_latent_nn_distance": np.nan,
                    "feasible_latent_nn_count": 0,
                    "feasible_lof_log_scale": np.nan,
                    "feasible_lof_score_cf": np.nan,
                    "feasible_lof_score_original": np.nan,
                    "feasible_lof_count": 0,
                    "lof_log_scale": np.nan,
                    "lof_score_cf": np.nan,
                    "lof_score_original": np.nan,
                    "lof_count": 0,
                    "constraint_violation_rate": 0.0,
                    "valid_constraint_violation_rate": np.nan,
                }
            ]
        )

    if enforce_constraints:
        constraint_valid = validate_constraints_for_evaluation(cfs, originals_by_id, info)
    else:
        constraint_valid = np.ones(generated_cf_count, dtype=bool)
    prob_valid = cfs["target_prob"].astype(float).to_numpy() > threshold
    valid_mask = prob_valid
    feasible_mask = prob_valid & constraint_valid
    valid_cfs = cfs.loc[valid_mask].reset_index(drop=True)
    feasible_cfs = cfs.loc[feasible_mask].reset_index(drop=True)

    metrics_df = _compute_row_level_change_metrics(
        valid_cfs, originals_by_id, feature_names, num_cols, cat_cols, info
    )
    feasible_metrics_df = _compute_row_level_change_metrics(
        feasible_cfs, originals_by_id, feature_names, num_cols, cat_cols, info
    )
    valid_cf_count = len(valid_cfs)
    valid_source_count = valid_cfs["id"].nunique()
    feasible_cf_count = len(feasible_cfs)
    feasible_source_count = feasible_cfs["id"].nunique()
    diversity_metrics = compute_diversity_metrics(
        valid_cfs, feature_names, num_cols, cat_cols, info
    )
    dpp_diversity_metrics = compute_dpp_diversity_metrics(
        valid_cfs, feature_names, num_cols, cat_cols, info
    )
    feasible_diversity_metrics = compute_diversity_metrics(
        feasible_cfs, feature_names, num_cols, cat_cols, info
    )
    feasible_dpp_diversity_metrics = compute_dpp_diversity_metrics(
        feasible_cfs, feature_names, num_cols, cat_cols, info
    )
    raw_diversity_metrics = compute_diversity_metrics(
        cfs, feature_names, num_cols, cat_cols, info
    )
    raw_dpp_diversity_metrics = compute_dpp_diversity_metrics(
        cfs, feature_names, num_cols, cat_cols, info
    )
    latent_metrics = _empty_latent_metrics()
    feasible_latent_metrics = _empty_latent_metrics()
    raw_latent_metrics = _empty_latent_metrics()
    if cf_latents is not None and reference_latents is not None:
        cf_latents = np.asarray(cf_latents)
        if len(cf_latents) == generated_cf_count:
            raw_latent_metrics = compute_nearest_neighbor_distance(
                cf_latents, reference_latents
            )
            latent_metrics = compute_nearest_neighbor_distance(
                cf_latents[valid_mask], reference_latents
            )
            feasible_latent_metrics = compute_nearest_neighbor_distance(
                cf_latents[feasible_mask], reference_latents
            )
    lof_metrics = compute_lof_log_scale_metrics(
        valid_cfs, originals_by_id, train_df, info
    )
    feasible_lof_metrics = compute_lof_log_scale_metrics(
        feasible_cfs, originals_by_id, train_df, info
    )
    valid_constraint_violation_rate = (
        float((~constraint_valid[valid_mask]).mean()) if valid_cf_count else np.nan
    )

    return pd.DataFrame(
        [
            {
                "attempted_count": attempted_count,
                "generated_cf_count": generated_cf_count,
                "expected_cf_count": expected_cf_count,
                "valid_cf_count": valid_cf_count,
                "valid_source_count": valid_source_count,
                "coverage": valid_source_count / attempted_count if attempted_count else 0.0,
                "validity": valid_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "validity_actual": valid_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "validity_generated": valid_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "validity_expected": valid_cf_count / expected_cf_count if expected_cf_count else 0.0,
                "mean_target_prob": float(valid_cfs["target_prob"].mean()) if valid_cf_count else np.nan,
                "sparsity": _mean_metric(metrics_df, "sparsity", valid_cf_count),
                "sparsity_cont": _mean_metric(metrics_df, "sparsity_cont", valid_cf_count),
                "sparsity_cat": _mean_metric(metrics_df, "sparsity_cat", valid_cf_count),
                "proximity": _mean_metric(metrics_df, "proximity", valid_cf_count),
                "proximity_cont": _mean_metric(metrics_df, "proximity_cont", valid_cf_count),
                "proximity_cat": _mean_metric(metrics_df, "proximity_cat", valid_cf_count),
                "diversity": diversity_metrics["diversity"],
                "diversity_cont": diversity_metrics["diversity_cont"],
                "diversity_cat": diversity_metrics["diversity_cat"],
                "diversity_pair_count": diversity_metrics["diversity_pair_count"],
                "dpp_diversity": dpp_diversity_metrics["dpp_diversity"],
                "log_dpp_diversity": dpp_diversity_metrics["log_dpp_diversity"],
                "dpp_pair_count": dpp_diversity_metrics["dpp_pair_count"],
                "raw_diversity": raw_diversity_metrics["diversity"],
                "raw_diversity_cont": raw_diversity_metrics["diversity_cont"],
                "raw_diversity_cat": raw_diversity_metrics["diversity_cat"],
                "raw_diversity_pair_count": raw_diversity_metrics[
                    "diversity_pair_count"
                ],
                "raw_dpp_diversity": raw_dpp_diversity_metrics["dpp_diversity"],
                "raw_log_dpp_diversity": raw_dpp_diversity_metrics[
                    "log_dpp_diversity"
                ],
                "raw_dpp_pair_count": raw_dpp_diversity_metrics["dpp_pair_count"],
                "latent_nn_distance": latent_metrics["latent_nn_distance"],
                "latent_nn_count": latent_metrics["latent_nn_count"],
                "raw_latent_nn_distance": raw_latent_metrics["latent_nn_distance"],
                "raw_latent_nn_count": raw_latent_metrics["latent_nn_count"],
                "feasible_cf_count": feasible_cf_count,
                "feasible_source_count": feasible_source_count,
                "feasible_coverage": feasible_source_count / attempted_count if attempted_count else 0.0,
                "feasible_validity": feasible_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "feasible_validity_actual": feasible_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "feasible_validity_generated": feasible_cf_count / generated_cf_count if generated_cf_count else 0.0,
                "feasible_validity_expected": feasible_cf_count / expected_cf_count if expected_cf_count else 0.0,
                "feasible_mean_target_prob": float(feasible_cfs["target_prob"].mean()) if feasible_cf_count else np.nan,
                "feasible_sparsity": _mean_metric(feasible_metrics_df, "sparsity", feasible_cf_count),
                "feasible_sparsity_cont": _mean_metric(feasible_metrics_df, "sparsity_cont", feasible_cf_count),
                "feasible_sparsity_cat": _mean_metric(feasible_metrics_df, "sparsity_cat", feasible_cf_count),
                "feasible_proximity": _mean_metric(feasible_metrics_df, "proximity", feasible_cf_count),
                "feasible_proximity_cont": _mean_metric(feasible_metrics_df, "proximity_cont", feasible_cf_count),
                "feasible_proximity_cat": _mean_metric(feasible_metrics_df, "proximity_cat", feasible_cf_count),
                "feasible_diversity": feasible_diversity_metrics["diversity"],
                "feasible_diversity_cont": feasible_diversity_metrics["diversity_cont"],
                "feasible_diversity_cat": feasible_diversity_metrics["diversity_cat"],
                "feasible_diversity_pair_count": feasible_diversity_metrics["diversity_pair_count"],
                "feasible_dpp_diversity": feasible_dpp_diversity_metrics["dpp_diversity"],
                "feasible_log_dpp_diversity": feasible_dpp_diversity_metrics["log_dpp_diversity"],
                "feasible_dpp_pair_count": feasible_dpp_diversity_metrics["dpp_pair_count"],
                "feasible_latent_nn_distance": feasible_latent_metrics["latent_nn_distance"],
                "feasible_latent_nn_count": feasible_latent_metrics["latent_nn_count"],
                "feasible_lof_log_scale": feasible_lof_metrics["lof_log_scale"],
                "feasible_lof_score_cf": feasible_lof_metrics["lof_score_cf"],
                "feasible_lof_score_original": feasible_lof_metrics["lof_score_original"],
                "feasible_lof_count": feasible_lof_metrics["lof_count"],
                "lof_log_scale": lof_metrics["lof_log_scale"],
                "lof_score_cf": lof_metrics["lof_score_cf"],
                "lof_score_original": lof_metrics["lof_score_original"],
                "lof_count": lof_metrics["lof_count"],
                "constraint_violation_rate": float((~constraint_valid).mean()) if generated_cf_count else 0.0,
                "valid_constraint_violation_rate": valid_constraint_violation_rate,
            }
        ]
    )


def build_feature_change_summary(cfs, originals, info):
    feature_names = _feature_names(info)
    num_cols, cat_cols = _num_cat_cols(info)
    num_cols = set(num_cols)
    cat_cols = set(cat_cols)

    columns = [
        "feature",
        "feature_type",
        "changed_count",
        "changed_rate",
        "mean_abs_change",
        "max_abs_change",
        "top_transition",
        "top_transition_count",
    ]
    if cfs.empty or originals.empty:
        return pd.DataFrame(columns=columns)

    merged = cfs.merge(originals, on="id", suffixes=("_cf", "_orig"))
    total = len(merged)
    rows = []

    for feature in feature_names:
        cf_col = f"{feature}_cf"
        orig_col = f"{feature}_orig"
        if cf_col not in merged.columns or orig_col not in merged.columns:
            continue

        if feature in num_cols:
            cf_values = merged[cf_col].astype(float)
            orig_values = merged[orig_col].astype(float)
            abs_change = (cf_values - orig_values).abs()
            changed = abs_change > 1e-8
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numerical",
                    "changed_count": int(changed.sum()),
                    "changed_rate": float(changed.mean()) if total else 0.0,
                    "mean_abs_change": float(abs_change.mean()) if total else 0.0,
                    "max_abs_change": float(abs_change.max()) if total else 0.0,
                    "top_transition": "",
                    "top_transition_count": 0,
                }
            )
        elif feature in cat_cols:
            cf_values = merged[cf_col]
            orig_values = merged[orig_col]
            changed = cf_values != orig_values
            changed_pairs = pd.DataFrame(
                {"orig": orig_values[changed], "cf": cf_values[changed]}
            ).astype(str)
            if changed_pairs.empty:
                top_transition = ""
                top_transition_count = 0
            else:
                transitions = (
                    changed_pairs["orig"] + " -> " + changed_pairs["cf"]
                ).value_counts()
                top_transition = transitions.index[0]
                top_transition_count = int(transitions.iloc[0])
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "categorical",
                    "changed_count": int(changed.sum()),
                    "changed_rate": float(changed.mean()) if total else 0.0,
                    "mean_abs_change": np.nan,
                    "max_abs_change": np.nan,
                    "top_transition": top_transition,
                    "top_transition_count": top_transition_count,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def load_latent_evaluation_arrays(save_path, method, args):
    if method != "mdpcf":
        return None, None

    cf_latents_path = os.path.join(save_path, "cf_latents.csv")
    if not os.path.isfile(cf_latents_path):
        return None, None

    output_file_name_csvae = get_name_form_args(args)
    train_z_path = os.path.join(
        "mdpcf", "csvae", "ckpt", args.dataname, output_file_name_csvae, "train_z.npy"
    )
    if not os.path.isfile(train_z_path):
        return None, None

    cf_latents = pd.read_csv(cf_latents_path, index_col=False)
    latent_cols = [col for col in cf_latents.columns if col.startswith("z_")]
    if not latent_cols:
        return None, None

    reference_latents = np.load(train_z_path)
    return cf_latents[latent_cols].to_numpy(), reference_latents


def main(args):
    dataname = args.dataname
    method = args.method
    save_path = args.save_path
    num_samples = args.num_samples
    source_start = getattr(args, "source_start", 0)
    pyod = args.pyod

    # validity = args.validity
    # proximity = args.proximity
    # sparsity = args.sparsity
    # latent_clf = args.latent_clf

    dice_method = args.dice_method
    dice_optimization = args.dice_optimization
    proximity_weight_input = args.proximity_weight_input
    proximity_weight_latent = args.proximity_weight_latent
    proximity_latent_loss = args.proximity_latent_loss

    total_CFs = args.total_CFs
    get_stats = args.get_stats
    get_changed = args.get_changed

    if method == "tabcf":
        optimization_parameters = get_tabcf_original_optim_params_form_args(args)
    else:
        optimization_parameters = get_optim_params_form_args(args)

    data_dir = f"data/{dataname}"
    info_path = f"data/{dataname}/info.json"
    train_set_path = f"data/{dataname}/train.csv"

    output_folder_name = dataname

    if method == "dice":
        output_folder_name = f"{dataname}/DiCE/{dice_method}/{optimization_parameters}"
    elif method == "mdpcf":
        output_file_name_csvae = get_name_form_args(args)
        output_folder_name = f"{dataname}/MDPCF/{output_file_name_csvae}/{dice_method}/{optimization_parameters}"
    elif method == "tabcf":
        output_file_name_vae = get_tabcf_original_name_form_args(args)
        output_folder_name = f"{dataname}/TABCF/{output_file_name_vae}/{dice_method}/{optimization_parameters}"
    elif method == "revise":
        output_folder_name = f"{dataname}/Revise"
    elif method == "cchvae":
        output_folder_name = f"{dataname}/CCHVAE"
    elif method == "dicoflex":
        output_folder_name = (
            f"{dataname}/DiCoFlex/"
            f"epochs[{args.dicoflex_epochs}]_p[{args.dicoflex_p_value}]_"
            f"temp[{args.dicoflex_temperature}]_nn[{args.dicoflex_n_nearest}]_"
            f"train[{args.dicoflex_train_samples}]"
        )
    elif method == "wachter":
        output_folder_name = f"{dataname}/Wachter"

    if num_samples > 0:
        output_folder_name = f"{output_folder_name}/{num_samples}_samples"
    else:
        output_folder_name = f"{output_folder_name}/all_samples"

    if method == "mdpcf" and source_start > 0:
        output_folder_name = f"{output_folder_name}_from_{source_start}"

    if total_CFs > 1:
        output_folder_name = f"{output_folder_name}_CFs_{total_CFs}"

    save_path = os.path.join(save_path, output_folder_name)

    # print("Evaluating:", output_folder_name)

    with open(info_path, "r") as f:
        info = json.load(f)

    target_class = info["target_class"]

    cfs = pd.read_csv(os.path.join(save_path, "cfs.csv"), index_col=False)

    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    column_names.remove(target_col)

    cfs = cfs[["id"] + column_names + ["target_prob"]]

    original_test_samples = pd.read_csv(
        os.path.join(save_path, "original_test_samples.csv"), index_col=False
    )

    cf_latents, reference_latents = load_latent_evaluation_arrays(
        save_path, method, args
    )
    train_df_for_summary = pd.read_csv(train_set_path)
    evaluation_summary = build_evaluation_summary(
        cfs,
        original_test_samples,
        info,
        total_CFs=total_CFs,
        cf_latents=cf_latents,
        reference_latents=reference_latents,
        train_df=train_df_for_summary,
        enforce_constraints=getattr(args, "enforce_constraints", False),
    )
    evaluation_summary.insert(0, "method", method)
    evaluation_summary.insert(0, "dataset", dataname)
    evaluation_summary.to_csv(
        os.path.join(save_path, "evaluation_summary.csv"), index=False
    )
    feature_change_summary = build_feature_change_summary(cfs, original_test_samples, info)
    feature_change_summary.to_csv(
        os.path.join(save_path, "feature_change_summary.csv"), index=False
    )

    print(output_folder_name)
    print(tabulate(evaluation_summary, headers="keys", tablefmt="pretty", showindex=False))

    if not get_stats and not get_changed:
        return

    # assert original_test_samples.iloc[:, info["target_col_idx"]].nunique()[0] == 1 or np.unique(original_test_samples[target_column])[0] != target_class, "Original test samples should all be from the negative class"

    up_arrow = "\u2191"
    down_arrow = "\u2193"

    cat_col_idx = info["cat_col_idx"]
    num_col_idx = info["num_col_idx"]
    num_cols = [column_names[x] for x in info["num_col_idx"]]
    cat_cols = [column_names[x] for x in info["cat_col_idx"]]
    num_max = {k: float(v) for k, v in info["num_max"].items()}
    num_min = {k: float(v) for k, v in info["num_min"].items()}

    train_df = train_df_for_summary
    # Initialize the StandardScaler
    scaler = StandardScaler()
    if len(num_cols) > 0:
        # Fit the scaler on the numerical features
        scaler.fit(train_df[num_cols])

    del train_df

    valid_cfs = []
    metrics = []
    all_correct_cfs = []

    feature_changes_count = dict(zip(column_names, np.zeros((len(column_names)))))

    for indx, row in original_test_samples.iterrows():

        original_sample = original_test_samples.iloc[indx : indx + 1]

        id = original_sample["id"].values[0]
        correct_cfs = cfs[(cfs["id"] == id) & (cfs["target_prob"] > 0.5)]

        if len(correct_cfs) == 0:
            continue

        valid_cfs.append(len(correct_cfs))

        correct_cfs_norm = correct_cfs.copy()
        correct_cfs_standard_norm = correct_cfs[num_cols].copy()
        original_sample_norm = original_sample.copy()

        original_sample_standard_norm = original_sample[num_cols].copy()

        if len(num_cols) > 0:

            correct_cfs_norm[num_cols] = correct_cfs_norm[num_cols].apply(
                lambda x: (x - num_min[x.name]) / (num_max[x.name] - num_min[x.name])
            )
            original_sample_norm[num_cols] = original_sample_norm[num_cols].apply(
                lambda x: (x - num_min[x.name]) / (num_max[x.name] - num_min[x.name])
            )

            correct_cfs_standard_norm = scaler.transform(correct_cfs_standard_norm)
            original_sample_standard_norm = scaler.transform(
                original_sample_standard_norm
            )[0]

        correct_cfs_norm = correct_cfs_norm[column_names].reset_index(drop=True)
        original_sample_norm = original_sample_norm[column_names].to_numpy()[0]

        sparsity_for_sample = []
        sparsity_for_sample_cat = []
        sparsity_for_sample_cont = []
        proximity_for_sample = []
        proximity_for_sample_cont = []

        for c_i, _ in correct_cfs_norm.iterrows():
            cf = correct_cfs_norm.iloc[c_i : c_i + 1].to_numpy()[0]
            if len(num_cols) > 0:
                cf_standard = correct_cfs_standard_norm[c_i]

            for feature in column_names:
                if (
                    correct_cfs.iloc[c_i : c_i + 1][feature].values
                    != original_sample[feature].values
                ):
                    feature_changes_count[feature] += 1

            sparsity_metric = l0_distance(cf, original_sample_norm) / len(cf)
            proximity_metric = gower_distance(cf, original_sample_norm, cat_col_idx)

            if len(num_cols) > 0:

                proximity_cont = np.absolute(cf_standard, original_sample_standard_norm)

            sparsity_metric_cat = l0_distance(
                cf[cat_col_idx], original_sample_norm[cat_col_idx]
            ) / len(cat_col_idx)

            if len(num_cols) > 0:

                sparsity_metric_cont = l0_distance(
                    cf[num_col_idx], original_sample_norm[num_col_idx]
                ) / len(num_col_idx)

            sparsity_for_sample.append(sparsity_metric)
            sparsity_for_sample_cat.append(sparsity_metric_cat)
            if len(num_cols) > 0:

                sparsity_for_sample_cont.append(sparsity_metric_cont)

            proximity_for_sample.append(proximity_metric)
            if len(num_cols) > 0:
                proximity_for_sample_cont.append(proximity_cont)

            all_correct_cfs.append(correct_cfs.iloc[c_i : c_i + 1])

        mean_sparsity = np.mean(sparsity_for_sample)
        mean_sparsity_cat = np.mean(sparsity_for_sample_cat)
        mean_proximity = np.mean(proximity_for_sample)

        if len(num_cols) > 0:
            mean_sparsity_cont = np.mean(sparsity_for_sample_cont)
            mean_proximity_cont = np.mean(proximity_for_sample_cont)

        if len(num_cols) > 0:

            metrics.append(
                [
                    mean_sparsity,
                    mean_proximity,
                    mean_sparsity_cat,
                    mean_sparsity_cont,
                    mean_proximity_cont,
                ]
            )
        else:
            metrics.append([mean_sparsity, mean_proximity, mean_sparsity_cat])

    average_metrics = np.mean(metrics, axis=0)

    if len(num_cols) > 0:

        mean_df = pd.DataFrame(
            [average_metrics],
            columns=[
                f"Sparsity {down_arrow}",
                f"Proximity (Gower) {down_arrow}",
                f"Sparsity cat {down_arrow}",
                f"Sparsity cont {down_arrow}",
                f"Prox cont {down_arrow}",
            ],
        )
    else:
        mean_df = pd.DataFrame(
            [average_metrics],
            columns=[
                f"Sparsity {down_arrow}",
                f"Proximity (Gower) {down_arrow}",
                f"Sparsity cat {down_arrow}",
            ],
        )

    len_cf_found = sum(valid_cfs)
    len_set = len(cfs)

    mean_df[f"Validity black box {up_arrow}"] = len_cf_found / len_set if len_set else 0.0

    mean_df.index = [method]

    train_set_path = f"data/{dataname}/train.csv"

    stats_df = pd.DataFrame()
    stats_df_target = pd.DataFrame()
    stats_df_negative = pd.DataFrame()
    outlier_df = pd.DataFrame()

    all_correct_cfs = pd.concat(all_correct_cfs)

    mean_df[f"Probability black box {up_arrow}"] = np.mean(
        all_correct_cfs["target_prob"]
    )

    all_correct_cfs = all_correct_cfs.drop(["id", "target_prob"], axis=1)
    print(tabulate(mean_df, headers="keys", tablefmt="pretty"))

    mean_df.to_csv(os.path.join(save_path, "results.csv"))

    if get_stats:

        stats_all_data = get_plausibility_stats(
            all_correct_cfs, train_set_path, info, data_dir, pyod=pyod
        )
        stats_positive_data = get_plausibility_stats(
            all_correct_cfs, train_set_path, info, data_dir, target_only=True, pyod=pyod
        )
        stats_negative_data = get_plausibility_stats(
            all_correct_cfs,
            train_set_path,
            info,
            data_dir,
            negative_only=True,
            pyod=pyod,
        )

        stats_df[f"Column-wise density (to train data) {up_arrow}"] = [
            stats_all_data[0]
        ]
        stats_df[f"Pair-wise column correlation (to train data) {up_arrow}"] = [
            stats_all_data[1]
        ]

        if pyod:
            outlier_df[f"Inlier probability (on train data) {up_arrow}"] = [
                stats_all_data[2]
            ]
        else:
            outlier_df[f"Anomaly scores (on train data) {up_arrow}"] = [
                stats_all_data[2]
            ]

        stats_df_target[
            f"Column-wise density (to positive class train data) {up_arrow}"
        ] = [stats_positive_data[0]]
        stats_df_target[
            f"Pair-wise column correlation (to positive class train data) {up_arrow}"
        ] = [stats_positive_data[1]]

        stats_df_negative[
            f"Column-wise density (to negative class train data) {up_arrow}"
        ] = [stats_negative_data[0]]
        stats_df_negative[
            f"Pair-wise column correlation (to negative class train data) {up_arrow}"
        ] = [stats_negative_data[1]]

        if pyod:
            outlier_df[
                f"Inlier probability (on positive class train data) {up_arrow}"
            ] = [stats_positive_data[2]]
        else:
            outlier_df[f"Anomaly scores (on positive class train data) {up_arrow}"] = [
                stats_positive_data[2]
            ]

        if pyod:
            outlier_df[
                f"Inlier probability (on negative class train data) {up_arrow}"
            ] = [stats_negative_data[2]]
        else:
            outlier_df[f"Anomaly scores (on negative class train data) {up_arrow}"] = [
                stats_negative_data[2]
            ]

        stats_df.index = [method]
        stats_df_target.index = [method]
        stats_df_negative.index = [method]
        outlier_df.index = [method]

        print(tabulate(stats_df, headers="keys", tablefmt="pretty"))
        print(tabulate(stats_df_target, headers="keys", tablefmt="pretty"))
        print(tabulate(stats_df_negative, headers="keys", tablefmt="pretty"))
        print(tabulate(outlier_df, headers="keys", tablefmt="pretty"))

    total_changes = sum(feature_changes_count.values())
    normalized_counter = {
        key: value / total_changes for key, value in feature_changes_count.items()
    }

    # Compute sums
    num_sum = sum(normalized_counter[col] for col in num_cols)
    cat_sum = sum(normalized_counter[col] for col in cat_cols)

    # Add new items to the dictionary
    normalized_counter["num_cols_sum"] = num_sum
    normalized_counter["cat_cols_sum"] = cat_sum
    if get_changed:
        print(normalized_counter)
        with open(os.path.join(save_path, "feature_changes.json"), "w") as file:
            json.dump(
                normalized_counter, file, indent=4
            )  # 'indent=4' is optional, it makes the file more readable


if __name__ == "__main__":
    from utils import get_args, prepare_runtime_args

    args = prepare_runtime_args(get_args())
    main(args)
