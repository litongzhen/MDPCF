import torch
import warnings
import time

from tqdm import tqdm

from utils_train import (
    preprocess,
    get_name_form_args,
    get_black_box_model_path,
    save_run_config,
    get_optim_params_form_args,
)

from mdpcf.csvae.model import BBMLPCLF
from mdpcf.checkpoints import (
    aggregate_sampling_stats,
    is_resumable_sampling_directory,
    load_completed_sources,
    merge_source_shards,
    write_source_checkpoint,
)
from mdpcf.latent_utils import (
    get_input_generate,
    recover_data,
    generate_equiangular_unit_vectors,
)
import numpy as np
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")


def select_source_indices(
    test_predictions,
    target_class_index,
    num_samples,
    source_start=0,
):
    if source_start < 0:
        raise ValueError("source_start must be non-negative")

    eligible_indices = np.where(test_predictions != target_class_index)[0]
    if num_samples > 0:
        selected_indices = eligible_indices[
            source_start : source_start + num_samples
        ]
        if len(selected_indices) != num_samples:
            raise ValueError(
                f"Requested {num_samples} source instances from offset "
                f"{source_start}, but only {len(selected_indices)} are available "
                f"from {len(eligible_indices)} eligible test instances."
            )
        return selected_indices

    return eligible_indices[source_start:]


def build_sample_tag(num_samples, total_CFs, source_start=0):
    sample_tag = f"{num_samples}_samples" if num_samples > 0 else "all_samples"
    if source_start > 0:
        sample_tag = f"{sample_tag}_from_{source_start}"
    if total_CFs > 1:
        sample_tag = f"{sample_tag}_CFs_{total_CFs}"
    return sample_tag


def commit_source_sampling_checkpoint(
    checkpoint_root,
    source_id,
    total_sources,
    stats_before_source,
    sampling_stats,
    shards,
):
    source_stats = {
        key: sampling_stats[key] - stats_before_source[key]
        for key in sampling_stats
        if key != "attempted_count"
    }
    source_stats["attempted_count"] = 1
    write_source_checkpoint(checkpoint_root, source_id, source_stats, shards)
    print(f"Sampling checkpoint: {source_id + 1}/{total_sources}")


def build_encoded_feature_groups(info):
    continuous_indexes = list(range(len(info["num_col_idx"])))
    categorical_slices = []

    start = len(info["num_col_idx"])
    for cat_col_idx in info["cat_col_idx"]:
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        width = len(info["column_domain"][col_name])
        categorical_slices.append(slice(start, start + width))
        start += width

    return continuous_indexes, categorical_slices


def build_actionability_masks(info, encoded_dim):
    immutable_cols = set(info.get("immutable", []))
    monotone_increase_cols = set(info.get("monotone_increase", []))
    monotone_decrease_cols = set(info.get("monotone_decrease", []))

    immutable_mask = np.zeros(encoded_dim, dtype=bool)
    monotone_increase_mask = np.zeros(encoded_dim, dtype=bool)
    monotone_decrease_mask = np.zeros(encoded_dim, dtype=bool)

    for local_idx, num_col_idx in enumerate(info["num_col_idx"]):
        col_name = info["idx_name_mapping"][str(num_col_idx)]
        immutable_mask[local_idx] = col_name in immutable_cols
        monotone_increase_mask[local_idx] = col_name in monotone_increase_cols
        monotone_decrease_mask[local_idx] = col_name in monotone_decrease_cols

    start = len(info["num_col_idx"])
    for cat_col_idx in info["cat_col_idx"]:
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        cat_width = len(info["column_domain"][col_name])
        if col_name in immutable_cols:
            immutable_mask[start : start + cat_width] = True
        start += cat_width

    return immutable_mask, monotone_increase_mask, monotone_decrease_mask


def apply_integer_constraints(df, info):
    for col_name in info.get("integer", []):
        if col_name not in df.columns:
            continue
        df[col_name] = df[col_name].round()
        if col_name in info.get("column_domain", {}):
            lower, upper = info["column_domain"][col_name]
            df[col_name] = df[col_name].clip(lower=lower, upper=upper)
        df[col_name] = df[col_name].astype(int)
    return df


def apply_numerical_domain_constraints(df, info):
    for num_col_idx in info["num_col_idx"]:
        col_name = info["idx_name_mapping"][str(num_col_idx)]
        if col_name not in df.columns:
            continue
        if col_name not in info.get("column_domain", {}):
            continue
        lower, upper = info["column_domain"][col_name]
        df[col_name] = df[col_name].astype(float).clip(lower=lower, upper=upper)
    return df


def attach_categorical_value_orders(info, cat_inverse):
    value_orders = {}
    n_cat = len(info.get("cat_col_idx", []))
    if n_cat == 0 or cat_inverse is None:
        info["cat_value_orders"] = value_orders
        return info

    for local_idx, cat_col_idx in enumerate(info["cat_col_idx"]):
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        width = len(info["column_domain"][col_name])
        encoded = np.zeros((width, n_cat), dtype=int)
        encoded[:, local_idx] = np.arange(width)
        decoded = cat_inverse(encoded)
        value_orders[col_name] = [row[local_idx] for row in decoded]

    info["cat_value_orders"] = value_orders
    return info


def get_categorical_value_order(info, col_name):
    return list(info.get("cat_value_orders", {}).get(
        col_name, info["column_domain"][col_name]
    ))


def encode_raw_features_for_black_box(df, info):
    encoded_parts = []

    for num_col_idx in info["num_col_idx"]:
        col_name = info["idx_name_mapping"][str(num_col_idx)]
        col = df[col_name].astype(float).to_numpy()
        min_value = float(info["num_min"][col_name])
        max_value = float(info["num_max"][col_name])
        denom = max(max_value - min_value, 1e-12)
        encoded_parts.append(((col - min_value) / denom).reshape(-1, 1))

    for cat_col_idx in info["cat_col_idx"]:
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        domain = get_categorical_value_order(info, col_name)
        values = df[col_name].to_numpy()
        one_hot = np.zeros((len(df), len(domain)), dtype=np.float32)
        domain_index = {value: idx for idx, value in enumerate(domain)}
        for row_idx, value in enumerate(values):
            if value in domain_index:
                one_hot[row_idx, domain_index[value]] = 1.0
        encoded_parts.append(one_hot)

    return np.concatenate(encoded_parts, axis=1).astype(np.float32)


def recover_data_from_encoded(encoded_data, info, num_inverse, cat_inverse):
    encoded_data = np.asarray(encoded_data, dtype=np.float32)
    d_numerical = len(info["num_col_idx"])

    syn_num = encoded_data[:, :d_numerical]
    syn_cat = []

    start = d_numerical
    for cat_col_idx in info["cat_col_idx"]:
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        width = len(info["column_domain"][col_name])
        syn_cat.append(encoded_data[:, start : start + width].argmax(axis=1))
        start += width

    syn_num = num_inverse(syn_num)
    if len(syn_cat) > 0:
        syn_cat = cat_inverse(np.stack(syn_cat, axis=1))
    else:
        syn_cat = np.empty((encoded_data.shape[0], 0))

    return recover_data(syn_num, syn_cat, info)


def apply_raw_actionability_constraints(df, original_row, info):
    for immutable_col in info.get("immutable", []):
        if immutable_col in df.columns:
            df[immutable_col] = original_row[immutable_col]

    for col_name in info.get("monotone_increase", []):
        if col_name in df.columns:
            df[col_name] = np.maximum(
                df[col_name].astype(float), float(original_row[col_name])
            )

    for col_name in info.get("monotone_decrease", []):
        if col_name in df.columns:
            df[col_name] = np.minimum(
                df[col_name].astype(float), float(original_row[col_name])
            )

    ordinal = info.get("ordinal", {})
    for col_name in info.get("ordinal_monotone_increase", []):
        if col_name not in df.columns or col_name not in ordinal:
            continue
        ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
        original_value = original_row[col_name]
        if original_value not in ranks:
            continue
        original_rank = ranks[original_value]
        df[col_name] = df[col_name].apply(
            lambda value: original_value
            if value in ranks and ranks[value] < original_rank
            else value
        )

    for col_name in info.get("ordinal_monotone_decrease", []):
        if col_name not in df.columns or col_name not in ordinal:
            continue
        ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
        original_value = original_row[col_name]
        if original_value not in ranks:
            continue
        original_rank = ranks[original_value]
        df[col_name] = df[col_name].apply(
            lambda value: original_value
            if value in ranks and ranks[value] > original_rank
            else value
        )

    return df


def _as_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def validate_counterfactual_constraints(cfs, original_row, info):
    valid = np.ones(len(cfs), dtype=bool)

    for immutable_col in info.get("immutable", []):
        if immutable_col in cfs.columns:
            valid &= cfs[immutable_col].to_numpy() == original_row[immutable_col]

    for col_name in info.get("monotone_increase", []):
        if col_name in cfs.columns:
            valid &= cfs[col_name].astype(float).to_numpy() >= float(original_row[col_name])

    for col_name in info.get("monotone_decrease", []):
        if col_name in cfs.columns:
            valid &= cfs[col_name].astype(float).to_numpy() <= float(original_row[col_name])

    for col_name in info.get("integer", []):
        if col_name in cfs.columns:
            values = cfs[col_name].astype(float).to_numpy()
            valid &= np.isclose(values, np.round(values))

    for num_col_idx in info["num_col_idx"]:
        col_name = info["idx_name_mapping"][str(num_col_idx)]
        if col_name not in cfs.columns or col_name not in info.get("column_domain", {}):
            continue
        lower, upper = info["column_domain"][col_name]
        values = cfs[col_name].astype(float).to_numpy()
        valid &= (values >= float(lower)) & (values <= float(upper))

    for cat_col_idx in info.get("cat_col_idx", []):
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        if col_name not in cfs.columns or col_name not in info.get("column_domain", {}):
            continue
        valid &= cfs[col_name].isin(info["column_domain"][col_name]).to_numpy()

    ordinal = info.get("ordinal", {})
    for col_name in info.get("ordinal_monotone_increase", []):
        if col_name not in cfs.columns or col_name not in ordinal:
            continue
        ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
        original_value = original_row[col_name]
        if original_value not in ranks:
            continue
        original_rank = ranks[original_value]
        cf_ranks = cfs[col_name].map(ranks).to_numpy()
        valid &= pd.notna(cf_ranks) & (cf_ranks >= original_rank)

    for col_name in info.get("ordinal_monotone_decrease", []):
        if col_name not in cfs.columns or col_name not in ordinal:
            continue
        ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
        original_value = original_row[col_name]
        if original_value not in ranks:
            continue
        original_rank = ranks[original_value]
        cf_ranks = cfs[col_name].map(ranks).to_numpy()
        valid &= pd.notna(cf_ranks) & (cf_ranks <= original_rank)

    return valid


def validate_counterfactuals(
    cfs,
    original_row,
    info,
    target_probs,
    threshold=0.5,
    enforce_constraints=True,
):
    target_valid = _as_numpy(target_probs) > threshold
    if not enforce_constraints:
        return target_valid
    constraints_valid = validate_counterfactual_constraints(cfs, original_row, info)
    return target_valid & constraints_valid


def _selection_feature_names(info, cfs):
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    ordered = [
        idx_name_mapping[idx]
        for idx in info.get("num_col_idx", []) + info.get("cat_col_idx", [])
        if idx in idx_name_mapping and idx_name_mapping[idx] in cfs.columns
    ]
    return ordered


def _selection_mixed_distance(row_a, row_b, feature_names, num_cols, cat_cols, info):
    distances = []
    for feature in feature_names:
        if feature in num_cols:
            num_min = float(info.get("num_min", {}).get(feature, 0.0))
            num_max = float(info.get("num_max", {}).get(feature, 1.0))
            denom = max(num_max - num_min, 1e-12)
            distances.append(abs(float(row_a[feature]) - float(row_b[feature])) / denom)
        elif feature in cat_cols:
            distances.append(float(row_a[feature] != row_b[feature]))
    return float(np.mean(distances)) if distances else 0.0


def _selection_cat_distance(row_a, row_b, cat_cols):
    distances = [
        float(row_a[feature] != row_b[feature])
        for feature in cat_cols
        if feature in row_a.index and feature in row_b.index
    ]
    return float(np.mean(distances)) if distances else 0.0


def _dpp_subset_score(distance_matrix, selected):
    if len(selected) <= 1:
        return 0.0
    sub_dist = distance_matrix[np.ix_(selected, selected)]
    kernel = 1.0 / (1.0 + sub_dist)
    kernel = kernel + 1e-6 * np.eye(len(selected))
    return float(np.linalg.det(kernel))


def fit_encoded_lof_selector(train_encoded, n_neighbors=20):
    train_encoded = np.asarray(train_encoded, dtype=float)
    if train_encoded.shape[0] < 2:
        return None
    effective_neighbors = min(n_neighbors, train_encoded.shape[0] - 1)
    if effective_neighbors < 1:
        return None
    lof = LocalOutlierFactor(n_neighbors=effective_neighbors, novelty=True)
    lof.fit(train_encoded)
    return lof


def score_raw_candidates_with_lof(cfs, info, lof_model):
    if lof_model is None or len(cfs) == 0:
        return None
    encoded = encode_raw_features_for_black_box(cfs, info)
    return -lof_model.score_samples(encoded)


def _normalize_selection_penalty(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    score_min = float(np.nanmin(scores))
    score_max = float(np.nanmax(scores))
    if not np.isfinite(score_min) or not np.isfinite(score_max):
        return np.zeros_like(scores, dtype=float)
    denom = score_max - score_min
    if denom < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - score_min) / denom


def _normalize_trajectory_metric(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Trajectory metrics must be finite")
    if maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def select_best_trajectory_snapshot(snapshot_metrics):
    if not snapshot_metrics:
        raise ValueError("At least one trajectory snapshot is required")
    diversity = _normalize_trajectory_metric(
        [snapshot["diversity"] for snapshot in snapshot_metrics]
    )
    lof = _normalize_trajectory_metric(
        [snapshot["lof"] for snapshot in snapshot_metrics]
    )
    target_prob = _normalize_trajectory_metric(
        [snapshot["target_prob"] for snapshot in snapshot_metrics]
    )
    proximity = _normalize_trajectory_metric(
        [snapshot["proximity"] for snapshot in snapshot_metrics]
    )
    scores = (
        0.35 * diversity
        + 0.35 * (1.0 - lof)
        + 0.15 * target_prob
        + 0.15 * (1.0 - proximity)
    )
    return int(np.argmax(scores)), scores.tolist()


def select_diverse_counterfactual_subset(
    cfs,
    latents,
    target_probs,
    info,
    total_CFs,
    selection_metric="mixed_distance",
    candidate_plausibility_scores=None,
):
    if len(cfs) <= total_CFs:
        return (
            cfs,
            latents,
            np.asarray(target_probs, dtype=float),
        )

    target_probs = np.asarray(target_probs, dtype=float)
    plausibility_penalty = None
    if candidate_plausibility_scores is not None:
        plausibility_penalty = _normalize_selection_penalty(
            candidate_plausibility_scores
        )
    feature_names = _selection_feature_names(info, cfs)
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    num_cols = {idx_name_mapping[idx] for idx in info.get("num_col_idx", [])}
    cat_cols = {idx_name_mapping[idx] for idx in info.get("cat_col_idx", [])}

    distance_matrix = np.zeros((len(cfs), len(cfs)), dtype=float)
    cat_distance_matrix = np.zeros((len(cfs), len(cfs)), dtype=float)
    for left_idx in range(len(cfs)):
        for right_idx in range(left_idx + 1, len(cfs)):
            distance = _selection_mixed_distance(
                cfs.iloc[left_idx],
                cfs.iloc[right_idx],
                feature_names,
                num_cols,
                cat_cols,
                info,
            )
            cat_distance = _selection_cat_distance(
                cfs.iloc[left_idx], cfs.iloc[right_idx], cat_cols
            )
            distance_matrix[left_idx, right_idx] = distance
            distance_matrix[right_idx, left_idx] = distance
            cat_distance_matrix[left_idx, right_idx] = cat_distance
            cat_distance_matrix[right_idx, left_idx] = cat_distance

    if selection_metric.startswith("lof_") and plausibility_penalty is not None:
        selected = [
            min(
                range(len(cfs)),
                key=lambda idx: (
                    float(plausibility_penalty[idx]),
                    -float(target_probs[idx]),
                ),
            )
        ]
    else:
        selected = [int(np.argmax(target_probs))]
    remaining = set(range(len(cfs))) - set(selected)

    while len(selected) < total_CFs and remaining:
        if selection_metric == "target_prob":
            best_idx = max(remaining, key=lambda idx: target_probs[idx])
        elif selection_metric == "dpp":
            best_idx = max(
                remaining,
                key=lambda idx: (
                    _dpp_subset_score(distance_matrix, selected + [idx]),
                    target_probs[idx],
                ),
            )
        elif selection_metric == "lof_dpp" and plausibility_penalty is not None:
            best_idx = max(
                remaining,
                key=lambda idx: (
                    -float(np.mean(plausibility_penalty[selected + [idx]])),
                    _dpp_subset_score(distance_matrix, selected + [idx]),
                    float(distance_matrix[idx, selected].min()),
                    target_probs[idx],
                ),
            )
        elif selection_metric == "lof_distance" and plausibility_penalty is not None:
            best_idx = max(
                remaining,
                key=lambda idx: (
                    -float(np.mean(plausibility_penalty[selected + [idx]])),
                    float(distance_matrix[idx, selected].min()),
                    float(distance_matrix[idx, selected].mean()),
                    target_probs[idx],
                ),
            )
        elif selection_metric == "category_first":
            best_idx = max(
                remaining,
                key=lambda idx: (
                    float(cat_distance_matrix[idx, selected].min()),
                    float(cat_distance_matrix[idx, selected].mean()),
                    float(distance_matrix[idx, selected].min()),
                    float(distance_matrix[idx, selected].mean()),
                    target_probs[idx],
                ),
            )
        else:
            best_idx = max(
                remaining,
                key=lambda idx: (
                    float(distance_matrix[idx, selected].min()),
                    float(distance_matrix[idx, selected].mean()),
                    target_probs[idx],
                ),
            )
        selected.append(int(best_idx))
        remaining.remove(best_idx)

    return (
        cfs.iloc[selected],
        latents.iloc[selected],
        target_probs[selected],
    )


def select_trajectory_quality_candidate_set(
    snapshots, info, original_row, total_CFs
):
    prepared = []
    feature_names = _selection_feature_names(info, snapshots[0]["cfs"])
    idx_name_mapping = {
        int(key): value for key, value in info["idx_name_mapping"].items()
    }
    num_cols = {idx_name_mapping[idx] for idx in info.get("num_col_idx", [])}
    cat_cols = {idx_name_mapping[idx] for idx in info.get("cat_col_idx", [])}

    for snapshot in snapshots:
        cfs = snapshot["cfs"].reset_index(drop=True)
        if len(cfs) < total_CFs:
            continue
        target_probs = np.asarray(snapshot["target_probs"], dtype=float)
        lof_scores = np.asarray(snapshot["lof_scores"], dtype=float)
        latent_frame = pd.DataFrame(
            _as_numpy(snapshot["latents"]).reshape(len(cfs), -1)
        )
        selected_cfs, _, selected_probs = select_diverse_counterfactual_subset(
            cfs,
            latent_frame,
            target_probs,
            info,
            total_CFs,
            selection_metric="lof_dpp",
            candidate_plausibility_scores=lof_scores,
        )
        positions = selected_cfs.index.to_numpy(dtype=int)
        selected_cfs = selected_cfs.reset_index(drop=True)
        diversity = compute_counterfactual_set_diversity(selected_cfs, info)
        proximities = [
            _selection_mixed_distance(
                selected_cfs.iloc[index],
                original_row,
                feature_names,
                num_cols,
                cat_cols,
                info,
            )
            for index in range(len(selected_cfs))
        ]
        prepared.append(
            {
                "step": int(snapshot["step"]),
                "cfs": selected_cfs,
                "latents": snapshot["latents"][positions].clone(),
                "decoded": snapshot["decoded"][positions].clone(),
                "target_probs": np.asarray(selected_probs, dtype=float),
                "diversity": float(diversity),
                "lof": float(lof_scores[positions].mean()),
                "target_prob": float(np.mean(selected_probs)),
                "proximity": float(np.mean(proximities)),
            }
        )

    if not prepared:
        raise ValueError("No trajectory snapshot contains enough valid candidates")
    best_index, scores = select_best_trajectory_snapshot(prepared)
    diagnostics = []
    for snapshot, score in zip(prepared, scores):
        diagnostics.append(
            {
                "step": snapshot["step"],
                "diversity": snapshot["diversity"],
                "lof": snapshot["lof"],
                "target_prob": snapshot["target_prob"],
                "proximity": snapshot["proximity"],
                "quality_score": float(score),
                "selected": int(len(diagnostics) == best_index),
            }
        )
    return prepared[best_index], diagnostics


def prepare_trajectory_quality_snapshots(
    trajectory_snapshots,
    info,
    num_inverse,
    cat_inverse,
    original_row,
    black_box_clf,
    lof_model,
    target_class_index,
    total_CFs,
    device,
    enforce_constraints=False,
):
    prepared = []
    idx_name_mapping = {
        int(key): value for key, value in info["idx_name_mapping"].items()
    }
    for snapshot in trajectory_snapshots:
        cfs = recover_data_from_encoded(
            snapshot["decoded"].numpy(), info, num_inverse, cat_inverse
        )
        cfs.rename(columns=idx_name_mapping, inplace=True)
        if enforce_constraints:
            apply_raw_actionability_constraints(cfs, original_row, info)
            apply_numerical_domain_constraints(cfs, info)
            apply_integer_constraints(cfs, info)
            apply_raw_actionability_constraints(cfs, original_row, info)
            apply_numerical_domain_constraints(cfs, info)
        rounded_encoded = encode_raw_features_for_black_box(cfs, info)
        with torch.no_grad():
            logits = black_box_clf(
                torch.tensor(rounded_encoded, dtype=torch.float32, device=device)
            )
            target_probs = (
                torch.softmax(logits, dim=-1)[:, target_class_index]
                .detach()
                .cpu()
                .numpy()
            )
        constraints_mask = (
            validate_counterfactual_constraints(cfs, original_row, info)
            if enforce_constraints
            else np.ones(len(cfs), dtype=bool)
        )
        valid_mask = (target_probs > 0.5) & constraints_mask
        if int(valid_mask.sum()) < total_CFs:
            continue
        valid_cfs = cfs.loc[valid_mask].reset_index(drop=True)
        lof_scores = score_raw_candidates_with_lof(valid_cfs, info, lof_model)
        if lof_scores is None:
            raise ValueError("Trajectory quality selection requires a fitted LOF model")
        prepared.append(
            {
                "step": snapshot["step"],
                "cfs": valid_cfs,
                "latents": snapshot["latents"][valid_mask].clone(),
                "decoded": snapshot["decoded"][valid_mask].clone(),
                "target_probs": target_probs[valid_mask],
                "lof_scores": lof_scores,
            }
        )
    return prepared


def compute_counterfactual_set_diversity(cfs, info):
    if len(cfs) < 2:
        return 0.0

    feature_names = _selection_feature_names(info, cfs)
    idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    num_cols = {idx_name_mapping[idx] for idx in info.get("num_col_idx", [])}
    cat_cols = {idx_name_mapping[idx] for idx in info.get("cat_col_idx", [])}

    distances = []
    for left_idx in range(len(cfs)):
        for right_idx in range(left_idx + 1, len(cfs)):
            distances.append(
                _selection_mixed_distance(
                    cfs.iloc[left_idx],
                    cfs.iloc[right_idx],
                    feature_names,
                    num_cols,
                    cat_cols,
                    info,
                )
            )
    return float(np.mean(distances)) if distances else 0.0


def apply_posthoc_categorical_diversity(
    cfs,
    original_row,
    info,
    predict_target_prob,
    threshold=0.5,
    enforce_constraints=True,
):
    diverse_cfs = cfs.copy().reset_index(drop=True)
    cat_features = [
        info["idx_name_mapping"][str(cat_col_idx)]
        for cat_col_idx in info.get("cat_col_idx", [])
        if info["idx_name_mapping"][str(cat_col_idx)] in diverse_cfs.columns
    ]
    rows = []

    for cf_idx in range(len(diverse_cfs)):
        accepted = 0
        attempted = 0
        for feature in cat_features:
            domain = info.get("column_domain", {}).get(feature, [])
            current_value = diverse_cfs.at[cf_idx, feature]
            best_candidate = None
            best_score = compute_counterfactual_set_diversity(diverse_cfs, info)
            best_prob = None

            for value in domain:
                if value == current_value:
                    continue
                attempted += 1
                candidate_set = diverse_cfs.copy()
                candidate_set.at[cf_idx, feature] = value
                if enforce_constraints:
                    apply_raw_actionability_constraints(candidate_set, original_row, info)
                    constraints_valid = validate_counterfactual_constraints(
                        candidate_set.loc[[cf_idx]], original_row, info
                    )[0]
                    if not constraints_valid:
                        continue
                candidate_prob = float(
                    predict_target_prob(candidate_set.loc[[cf_idx]])[0]
                )
                if candidate_prob <= threshold:
                    continue
                candidate_score = compute_counterfactual_set_diversity(
                    candidate_set, info
                )
                if candidate_score > best_score + 1e-12:
                    best_score = candidate_score
                    best_candidate = candidate_set.loc[[cf_idx]].copy()
                    best_prob = candidate_prob

            if best_candidate is not None:
                diverse_cfs.loc[cf_idx, best_candidate.columns] = best_candidate.iloc[0]
                accepted += 1

        final_prob = float(predict_target_prob(diverse_cfs.loc[[cf_idx]])[0])
        rows.append(
            {
                "candidate_index": cf_idx,
                "categorical_diversity_attempted": attempted,
                "categorical_diversity_accepted": accepted,
                "target_prob_after_diversity": final_prob,
            }
        )

    final_probs = predict_target_prob(diverse_cfs)
    return diverse_cfs, np.asarray(final_probs, dtype=float), pd.DataFrame(rows)


def count_changed_features(row, original_row, info):
    count = 0
    for num_col_idx in info.get("num_col_idx", []):
        col_name = info["idx_name_mapping"][str(num_col_idx)]
        if col_name in row.index and not np.isclose(
            float(row[col_name]), float(original_row[col_name])
        ):
            count += 1

    for cat_col_idx in info.get("cat_col_idx", []):
        col_name = info["idx_name_mapping"][str(cat_col_idx)]
        if col_name in row.index and row[col_name] != original_row[col_name]:
            count += 1

    return count


def apply_dice_posthoc_sparsity(
    cfs,
    original_row,
    info,
    predict_target_prob,
    threshold=0.5,
    continuous_features=None,
    enforce_constraints=True,
):
    sparse_cfs = cfs.copy().reset_index(drop=True)
    continuous_features = continuous_features or [
        info["idx_name_mapping"][str(num_col_idx)]
        for num_col_idx in info.get("num_col_idx", [])
    ]

    rows = []
    final_probs = []

    def is_candidate_valid(candidate):
        candidate_prob = float(predict_target_prob(candidate)[0])
        constraints_valid = (
            validate_counterfactual_constraints(candidate, original_row, info)[0]
            if enforce_constraints
            else True
        )
        return constraints_valid and candidate_prob > threshold

    def project_candidate(row, feature, value):
        candidate = row.copy()
        candidate.at[candidate.index[0], feature] = value
        if enforce_constraints:
            apply_raw_actionability_constraints(candidate, original_row, info)
            apply_numerical_domain_constraints(candidate, info)
            apply_integer_constraints(candidate, info)
        return candidate

    for cf_idx in range(len(sparse_cfs)):
        changed_before = count_changed_features(sparse_cfs.loc[cf_idx], original_row, info)
        prob_before = float(predict_target_prob(sparse_cfs.loc[[cf_idx]])[0])
        attempted = 0
        accepted = 0

        for feature in continuous_features:
            if feature not in sparse_cfs.columns or feature not in original_row.index:
                continue
            current_value = float(sparse_cfs.at[cf_idx, feature])
            original_value = float(original_row[feature])
            if np.isclose(current_value, original_value):
                continue

            attempted += 1
            current_row = sparse_cfs.loc[[cf_idx]].copy()
            candidate = project_candidate(current_row, feature, original_row[feature])
            candidate_valid = is_candidate_valid(candidate)

            if candidate_valid:
                sparse_cfs.loc[cf_idx, candidate.columns] = candidate.iloc[0]
                accepted += 1
                continue

            low = original_value
            high = current_value
            best = current_value
            for _ in range(25):
                midpoint = low + (high - low) / 2.0
                candidate = project_candidate(current_row, feature, midpoint)
                candidate_value = float(candidate.iloc[0][feature])
                if np.isclose(candidate_value, high):
                    break
                if is_candidate_valid(candidate):
                    best = candidate_value
                    high = candidate_value
                else:
                    low = candidate_value

            if not np.isclose(best, current_value):
                candidate = project_candidate(current_row, feature, best)
                sparse_cfs.loc[cf_idx, candidate.columns] = candidate.iloc[0]
                accepted += 1

        prob_after = float(predict_target_prob(sparse_cfs.loc[[cf_idx]])[0])
        final_probs.append(prob_after)
        rows.append(
            {
                "cf_index": cf_idx,
                "target_prob_before_sparsity": prob_before,
                "target_prob_after_sparsity": prob_after,
                "continuous_revert_attempted": attempted,
                "continuous_revert_accepted": accepted,
                "changed_features_before": changed_before,
                "changed_features_after": count_changed_features(
                    sparse_cfs.loc[cf_idx], original_row, info
                ),
            }
        )

    return sparse_cfs, np.asarray(final_probs, dtype=float), pd.DataFrame(rows)


def build_sampling_diagnostics(stats):
    attempted_count = int(stats.get("attempted_count", 0))
    optimized_valid_source_count = int(stats.get("optimized_valid_source_count", 0))
    optimized_valid_candidate_count = int(stats.get("optimized_valid_candidate_count", 0))
    post_projection_candidate_count = int(stats.get("post_projection_candidate_count", 0))
    post_projection_validity_count = int(stats.get("post_projection_validity_count", 0))
    post_projection_constraints_count = int(
        stats.get("post_projection_constraints_count", 0)
    )
    saved_cf_count = int(stats.get("saved_cf_count", 0))
    saved_source_count = int(stats.get("saved_source_count", 0))

    optimization_failed_count = attempted_count - optimized_valid_source_count
    filtered_by_validity = (
        post_projection_candidate_count - post_projection_validity_count
    )
    filtered_by_constraints = (
        post_projection_candidate_count - post_projection_constraints_count
    )

    return pd.DataFrame(
        [
            {
                "attempted_count": attempted_count,
                "optimized_valid_source_count": optimized_valid_source_count,
                "optimized_valid_candidate_count": optimized_valid_candidate_count,
                "post_projection_candidate_count": post_projection_candidate_count,
                "post_projection_validity_count": post_projection_validity_count,
                "post_projection_constraints_count": post_projection_constraints_count,
                "saved_cf_count": saved_cf_count,
                "saved_source_count": saved_source_count,
                "optimization_failed_count": max(optimization_failed_count, 0),
                "filtered_by_validity": max(filtered_by_validity, 0),
                "filtered_by_constraints": max(filtered_by_constraints, 0),
                "coverage": saved_source_count / attempted_count
                if attempted_count > 0
                else 0.0,
            }
        ]
    )


def build_projection_shift_summary(
    source_id,
    decoded_encoded,
    rounded_encoded,
    before_probs,
    after_probs,
    validity_mask,
    constraints_mask,
    info,
):
    decoded_np = _as_numpy(decoded_encoded).astype(np.float32)
    rounded_np = _as_numpy(rounded_encoded).astype(np.float32)
    before_probs_np = _as_numpy(before_probs).astype(float)
    after_probs_np = _as_numpy(after_probs).astype(float)
    validity_np = np.asarray(validity_mask, dtype=bool)
    constraints_np = np.asarray(constraints_mask, dtype=bool)

    d_numerical = len(info.get("num_col_idx", []))
    diff = np.abs(decoded_np - rounded_np)
    num_diff = diff[:, :d_numerical] if d_numerical > 0 else np.empty((len(diff), 0))
    cat_diff = diff[:, d_numerical:] if d_numerical < diff.shape[1] else np.empty((len(diff), 0))

    rows = []
    for candidate_index in range(decoded_np.shape[0]):
        rows.append(
            {
                "id": source_id,
                "candidate_index": candidate_index,
                "target_prob_before_projection": before_probs_np[candidate_index],
                "target_prob_after_projection": after_probs_np[candidate_index],
                "prob_drop": before_probs_np[candidate_index]
                - after_probs_np[candidate_index],
                "encoded_l1_shift": float(diff[candidate_index].sum()),
                "encoded_l2_shift": float(np.linalg.norm(diff[candidate_index], ord=2)),
                "num_l1_shift": float(num_diff[candidate_index].sum())
                if num_diff.shape[1] > 0
                else 0.0,
                "cat_l1_shift": float(cat_diff[candidate_index].sum())
                if cat_diff.shape[1] > 0
                else 0.0,
                "valid_after_projection": bool(validity_np[candidate_index]),
                "constraints_after_projection": bool(constraints_np[candidate_index]),
                "saved": bool(validity_np[candidate_index] and constraints_np[candidate_index]),
            }
        )
    return pd.DataFrame(rows)


def build_feature_change_summary(cfs, originals, info):
    feature_names = [
        info["idx_name_mapping"][str(col_idx)]
        for col_idx in info.get("num_col_idx", []) + info.get("cat_col_idx", [])
    ]
    numerical_features = {
        info["idx_name_mapping"][str(col_idx)] for col_idx in info.get("num_col_idx", [])
    }
    categorical_features = {
        info["idx_name_mapping"][str(col_idx)] for col_idx in info.get("cat_col_idx", [])
    }

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

        if feature in numerical_features:
            cf_values = merged[cf_col].astype(float)
            orig_values = merged[orig_col].astype(float)
            abs_change = (cf_values - orig_values).abs()
            changed = abs_change > 1e-8
            mean_abs_change = float(abs_change.mean()) if total > 0 else 0.0
            max_abs_change = float(abs_change.max()) if total > 0 else 0.0
            feature_type = "numerical"
            top_transition = ""
            top_transition_count = 0
        elif feature in categorical_features:
            cf_values = merged[cf_col]
            orig_values = merged[orig_col]
            changed = cf_values != orig_values
            changed_pairs = (
                pd.DataFrame({"orig": orig_values[changed], "cf": cf_values[changed]})
                .astype(str)
            )
            if changed_pairs.empty:
                top_transition = ""
                top_transition_count = 0
            else:
                pair_counts = changed_pairs.value_counts().reset_index(name="count")
                top_pair = pair_counts.iloc[0]
                top_transition = f"{top_pair['orig']} -> {top_pair['cf']}"
                top_transition_count = int(top_pair["count"])
            mean_abs_change = np.nan
            max_abs_change = np.nan
            feature_type = "categorical"
        else:
            continue

        changed_count = int(changed.sum())
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "changed_count": changed_count,
                "changed_rate": changed_count / total if total > 0 else 0.0,
                "mean_abs_change": mean_abs_change,
                "max_abs_change": max_abs_change,
                "top_transition": top_transition,
                "top_transition_count": top_transition_count,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def compute_num_feature_weights(encoded_train, continuous_indexes, eps=1e-4):
    if len(continuous_indexes) == 0:
        return torch.empty(0, dtype=torch.float32)

    train_num = encoded_train[:, continuous_indexes]
    median = np.median(train_num, axis=0)
    mad = np.median(np.abs(train_num - median), axis=0)
    mad = np.clip(mad, eps, None)
    return torch.tensor(1.0 / mad, dtype=torch.float32)


def compute_tabcf_input_proximity(
    decoded,
    original,
    continuous_indexes,
    categorical_slices,
    num_feature_weights,
):
    losses = []

    if len(continuous_indexes) > 0:
        idx = torch.tensor(continuous_indexes, dtype=torch.long, device=decoded.device)
        weights = num_feature_weights.to(decoded.device)
        num_diff = torch.abs(decoded[:, idx] - original[:, idx])
        losses.append((num_diff * weights).mean())

    if len(categorical_slices) > 0:
        cat_loss = torch.tensor(0.0, dtype=decoded.dtype, device=decoded.device)
        for cat_slice in categorical_slices:
            cat_loss = cat_loss + torch.abs(
                decoded[:, cat_slice] - original[:, cat_slice]
            ).mean()
        losses.append(cat_loss / len(categorical_slices))

    if len(losses) == 0:
        return torch.tensor(0.0, dtype=decoded.dtype, device=decoded.device)
    return sum(losses)


def compute_validity_loss(
    logits, target, target_class_index, validity_loss, validity_margin=1.0
):
    if validity_loss == "cross_entropy":
        return torch.nn.functional.cross_entropy(logits, target)

    if validity_loss != "hinge":
        raise ValueError(f"Unknown validity_loss: {validity_loss}")

    if logits.shape[-1] != 2:
        raise ValueError("hinge validity loss currently supports binary logits only")

    score = logits[:, 1] - logits[:, 0]
    z_sign = 1.0 if target_class_index == 1 else -1.0
    return torch.relu(validity_margin - z_sign * score).mean()


def compute_dpp_diversity(decoded, variant="det", eps=1e-6):
    if decoded.shape[0] <= 1:
        return torch.tensor(0.0, dtype=decoded.dtype, device=decoded.device)

    pairwise_dist = torch.cdist(decoded, decoded, p=1)
    kernel = 1.0 / (1.0 + pairwise_dist)
    kernel = kernel + eps * torch.eye(
        decoded.shape[0], dtype=decoded.dtype, device=decoded.device
    )
    if variant == "det":
        return -torch.linalg.det(kernel)
    if variant == "log":
        eigenvalues = torch.linalg.eigvalsh(kernel)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        return -torch.log(eigenvalues).sum()
    raise ValueError(f"Unknown DPP diversity variant: {variant}")


def compute_auto_latent_init_radius(
    train_z,
    scale=1.0,
    max_samples=5000,
    random_state=42,
):
    train_z = np.asarray(train_z, dtype=np.float32).reshape(len(train_z), -1)
    if len(train_z) < 2:
        raise ValueError("auto latent init radius requires at least two train latents")

    if max_samples and max_samples > 0 and len(train_z) > max_samples:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(train_z), size=max_samples, replace=False)
        train_z = train_z[sample_idx]

    diff = train_z[:, None, :] - train_z[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    radius = float(np.median(nearest) * scale)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Invalid auto latent init radius: {radius}")
    return radius


def resolve_latent_init_radius(args, train_z):
    strategy = getattr(args, "latent_init_radius_strategy", "fixed")
    if strategy == "fixed":
        return float(getattr(args, "latent_init_radius", 0.5))
    if strategy == "auto_nn":
        return compute_auto_latent_init_radius(
            train_z,
            scale=float(getattr(args, "latent_init_radius_scale", 1.0)),
            max_samples=int(getattr(args, "latent_init_radius_max_samples", 5000)),
        )
    raise ValueError(f"Unknown latent_init_radius_strategy: {strategy}")


def fit_latent_gmm_plausibility(
    train_z,
    n_components,
    max_samples,
    device,
    random_state=42,
    reg_covar=1e-6,
):
    train_z = np.asarray(train_z, dtype=np.float32).reshape(len(train_z), -1)
    if max_samples and max_samples > 0 and len(train_z) > max_samples:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(train_z), size=max_samples, replace=False)
        train_z = train_z[sample_idx]

    n_components = min(max(1, int(n_components)), len(train_z))
    # Collapsed latent dimensions can make a full-covariance component
    # numerically singular. Keep the full model as the default, but make the
    # plausibility term robust to this dataset/seed-specific failure.
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=reg_covar,
            random_state=random_state,
        )
        gmm.fit(train_z)
    except (np.linalg.LinAlgError, ValueError):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=max(float(reg_covar), 1e-4),
            random_state=random_state,
        )
        gmm.fit(train_z)

    covariances = gmm.covariances_.astype(np.float32)
    if gmm.covariance_type == "diag":
        covariances = np.array([np.diag(cov) for cov in covariances])
    precisions = np.linalg.inv(covariances).astype(np.float32)
    _, log_dets = np.linalg.slogdet(covariances)

    return {
        "means": torch.tensor(gmm.means_.astype(np.float32), device=device),
        "precisions": torch.tensor(precisions, device=device),
        "log_dets": torch.tensor(log_dets.astype(np.float32), device=device),
        "log_weights": torch.log(
            torch.tensor(gmm.weights_.astype(np.float32), device=device).clamp_min(1e-12)
        ),
    }


def compute_latent_gmm_plausibility_loss(z_candidates, gmm_params):
    if gmm_params is None:
        return torch.tensor(0.0, dtype=z_candidates.dtype, device=z_candidates.device)

    z_flat = z_candidates.reshape(z_candidates.shape[0], -1)
    means = gmm_params["means"].to(dtype=z_flat.dtype, device=z_flat.device)
    precisions = gmm_params["precisions"].to(dtype=z_flat.dtype, device=z_flat.device)
    log_dets = gmm_params["log_dets"].to(dtype=z_flat.dtype, device=z_flat.device)
    log_weights = gmm_params["log_weights"].to(dtype=z_flat.dtype, device=z_flat.device)

    diff = z_flat[:, None, :] - means[None, :, :]
    mahalanobis = torch.einsum("nkd,kde,nke->nk", diff, precisions, diff)
    dim = z_flat.shape[-1]
    log_prob = log_weights[None, :] - 0.5 * (
        dim * np.log(2.0 * np.pi) + log_dets[None, :] + mahalanobis
    )
    return -torch.logsumexp(log_prob, dim=1).mean()


class Decoder_Black_Box(torch.nn.Module):
    def __init__(
        self,
        black_box_clf,
        pre_decoder,
        info,
        categorical_straight_through=False,
    ):
        super(Decoder_Black_Box, self).__init__()
        self.black_box_clf = black_box_clf
        self.pre_decoder = pre_decoder
        self.info = info
        self.categorical_straight_through = categorical_straight_through
        self.fixed_w = None
        self.fixed_z = None
        self.immutable_mask = None
        self.monotone_increase_mask = None
        self.monotone_decrease_mask = None
        self.reference_values = None
        self.ordinal_constraints = self._build_ordinal_constraints(info)


    def _build_ordinal_constraints(self, info):
        ordinal = info.get("ordinal", {})
        monotone_increase = set(info.get("ordinal_monotone_increase", []))
        monotone_decrease = set(info.get("ordinal_monotone_decrease", []))
        constraints = []

        for cat_col_idx in info["cat_col_idx"]:
            col_name = info["idx_name_mapping"][str(cat_col_idx)]
            if col_name not in ordinal:
                constraints.append(None)
                continue

            if col_name in monotone_increase:
                direction = "increase"
            elif col_name in monotone_decrease:
                direction = "decrease"
            else:
                constraints.append(None)
                continue

            ranks = {value: rank for rank, value in enumerate(ordinal[col_name])}
            domain = get_categorical_value_order(info, col_name)
            domain_ranks = [ranks.get(value, -1) for value in domain]
            constraints.append(
                {
                    "col_name": col_name,
                    "direction": direction,
                    "domain_ranks": torch.tensor(domain_ranks, dtype=torch.long),
                }
            )

        return constraints


    def _get_reference_cat_probs(self, cat_index, batch_size, device):
        if self.reference_values is None:
            return None

        reference_values = self.reference_values.to(device)
        if reference_values.dim() == 1:
            reference_values = reference_values.unsqueeze(0)
        if reference_values.shape[0] != batch_size:
            reference_values = reference_values.repeat(batch_size, 1)

        start = len(self.info["num_col_idx"])
        for idx, cat_col_idx in enumerate(self.info["cat_col_idx"]):
            col_name = self.info["idx_name_mapping"][str(cat_col_idx)]
            width = len(self.info["column_domain"][col_name])
            if idx == cat_index:
                return reference_values[:, start : start + width]
            start += width
        return None


    def _mask_ordinal_logits(self, cat_pred, cat_index):
        if cat_index >= len(self.ordinal_constraints):
            return cat_pred
        constraint = self.ordinal_constraints[cat_index]
        if constraint is None:
            return cat_pred

        reference_cat = self._get_reference_cat_probs(
            cat_index, cat_pred.shape[0], cat_pred.device
        )
        if reference_cat is None:
            return cat_pred

        domain_ranks = constraint["domain_ranks"].to(cat_pred.device)
        ref_indexes = reference_cat.argmax(dim=-1)
        ref_ranks = domain_ranks[ref_indexes].unsqueeze(1)
        candidate_ranks = domain_ranks.unsqueeze(0)
        ranked = (ref_ranks >= 0) & (candidate_ranks >= 0)

        if constraint["direction"] == "increase":
            forbidden = ranked & (candidate_ranks < ref_ranks)
        else:
            forbidden = ranked & (candidate_ranks > ref_ranks)

        return cat_pred.masked_fill(forbidden, -1e9)


    def _process_cat_outputs(self, x_hat_cat):
        processed_cats = []
        for cat_index, cat_pred in enumerate(x_hat_cat):
            if cat_pred.dim() != 2:
                processed_cats.append(cat_pred)
                continue

            cat_pred = self._mask_ordinal_logits(cat_pred, cat_index)
            probs = torch.softmax(cat_pred, dim=-1)
            if self.categorical_straight_through:
                hard = torch.nn.functional.one_hot(
                    probs.argmax(dim=-1), num_classes=probs.shape[-1]
                ).to(dtype=probs.dtype)
                processed_cats.append(hard + probs - probs.detach())
            else:
                processed_cats.append(probs)
        return processed_cats


    def set_fixed_w(self, w):
        self.fixed_w = w.detach()


    def set_fixed_z(self, z):
        self.fixed_z = z.detach()


    def set_actionability_reference(
        self,
        immutable_mask,
        monotone_increase_mask,
        monotone_decrease_mask,
        reference_values,
    ):
        if not torch.is_tensor(immutable_mask):
            immutable_mask = torch.tensor(immutable_mask, dtype=torch.bool)
        if not torch.is_tensor(monotone_increase_mask):
            monotone_increase_mask = torch.tensor(
                monotone_increase_mask, dtype=torch.bool
            )
        if not torch.is_tensor(monotone_decrease_mask):
            monotone_decrease_mask = torch.tensor(
                monotone_decrease_mask, dtype=torch.bool
            )
        if not torch.is_tensor(reference_values):
            reference_values = torch.tensor(reference_values, dtype=torch.float32)
        self.immutable_mask = immutable_mask.detach()
        self.monotone_increase_mask = monotone_increase_mask.detach()
        self.monotone_decrease_mask = monotone_decrease_mask.detach()
        self.reference_values = reference_values.detach()


    def clear_actionability_reference(self):
        self.immutable_mask = None
        self.monotone_increase_mask = None
        self.monotone_decrease_mask = None
        self.reference_values = None


    def _apply_actionability_constraints(self, decoded):
        if self.immutable_mask is None or self.reference_values is None:
            return decoded
        immutable_mask = self.immutable_mask.to(decoded.device)
        monotone_increase_mask = self.monotone_increase_mask.to(decoded.device)
        monotone_decrease_mask = self.monotone_decrease_mask.to(decoded.device)
        reference_values = self.reference_values.to(decoded.device)

        masks = [immutable_mask, monotone_increase_mask, monotone_decrease_mask]
        masks = [mask.unsqueeze(0) if mask.dim() == 1 else mask for mask in masks]
        immutable_mask, monotone_increase_mask, monotone_decrease_mask = masks
        if reference_values.dim() == 1:
            reference_values = reference_values.unsqueeze(0)

        if immutable_mask.shape[0] != decoded.shape[0]:
            immutable_mask = immutable_mask.repeat(decoded.shape[0], 1)
        if monotone_increase_mask.shape[0] != decoded.shape[0]:
            monotone_increase_mask = monotone_increase_mask.repeat(decoded.shape[0], 1)
        if monotone_decrease_mask.shape[0] != decoded.shape[0]:
            monotone_decrease_mask = monotone_decrease_mask.repeat(decoded.shape[0], 1)
        if reference_values.shape[0] != decoded.shape[0]:
            reference_values = reference_values.repeat(decoded.shape[0], 1)

        constrained = torch.where(immutable_mask, reference_values, decoded)
        constrained = torch.where(
            monotone_increase_mask, torch.maximum(constrained, reference_values), constrained
        )
        constrained = torch.where(
            monotone_decrease_mask, torch.minimum(constrained, reference_values), constrained
        )
        return constrained


    def bb_only(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor
        # Pass concatenated output through pre_decoder
        output = self.black_box_clf(input_data)

        if original_input_dim == 1:
            return output[0]

        return output


    def decode_z(self, input_data):
        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        if self.fixed_w is None:
            raise ValueError("fixed_w must be set before decoding z")
        fixed_w = self.fixed_w.detach()
        
        if fixed_w.dim() == 1:
            fixed_w = fixed_w.unsqueeze(0)
            
        if fixed_w.shape[0] != input_data.shape[0]:
            fixed_w = fixed_w.repeat(input_data.shape[0], 1)
            
        x_hat_num, x_hat_cat = self.pre_decoder(input_data, fixed_w)
        processed_cats = self._process_cat_outputs(x_hat_cat)
        concatenated = torch.cat((x_hat_num, *processed_cats), dim=-1)
        return self._apply_actionability_constraints(concatenated)


    def decode_w(self, input_data):
        original_input_dim = input_data.dim()
        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)

        if self.fixed_z is None:
            raise ValueError("fixed_z must be set before decoding w")
        fixed_z = self.fixed_z.detach()
        if fixed_z.dim() == 1:
            fixed_z = fixed_z.unsqueeze(0)
        if fixed_z.shape[0] != input_data.shape[0]:
            fixed_z = fixed_z.repeat(input_data.shape[0], 1)

        x_hat_num, x_hat_cat = self.pre_decoder(fixed_z, input_data)
        processed_cats = self._process_cat_outputs(x_hat_cat)
        concatenated = torch.cat((x_hat_num, *processed_cats), dim=-1)
        return self._apply_actionability_constraints(concatenated)


    def forward(self, input_data):

        original_input_dim = input_data.dim()

        if original_input_dim == 1:
            input_data = input_data.unsqueeze(0)  # Unsqueezes the tensor

        if self.fixed_w is None:
            raise ValueError("fixed_w must be set before forward")
        fixed_w = self.fixed_w.detach()
        if fixed_w.dim() == 1:
            fixed_w = fixed_w.unsqueeze(0)
        if fixed_w.shape[0] != input_data.shape[0]:
            fixed_w = fixed_w.repeat(input_data.shape[0], 1)
        x_hat_num, x_hat_cat = self.pre_decoder(input_data, fixed_w)


        processed_cats = self._process_cat_outputs(x_hat_cat)
        concatenated = torch.cat((x_hat_num, *processed_cats), dim=-1)
        concatenated = self._apply_actionability_constraints(concatenated)

        output = self.black_box_clf(concatenated)

        if original_input_dim == 1:
            return output[0]

        return output


def initialize_latent_candidates(latent_center, total_cfs, radius, device):
    latent_dim = latent_center.shape[-1]
    directions = generate_equiangular_unit_vectors(total_cfs, d=latent_dim)
    directions = torch.tensor(directions, dtype=torch.float32, device=device)
    return (latent_center.unsqueeze(0) + radius * directions).clone().detach()


def resolve_fixed_decoder_context(
    latent_model, source_context, target_class_index, device
):
    if latent_model == "cvae":
        return torch.tensor([target_class_index], dtype=torch.long, device=device)
    return source_context


def resolve_latent_initialization_center(source, target_prior, target_prior_weight):
    weight = float(target_prior_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("target_prior_init_weight must be between 0 and 1")
    if target_prior is None or weight == 0.0:
        return source
    return source + weight * (target_prior - source)


def expand_target_prior_for_search(target_prior, search_latent_shape):
    target_prior = torch.as_tensor(target_prior)
    search_latent_shape = tuple(int(size) for size in search_latent_shape)
    search_size = int(np.prod(search_latent_shape))
    if target_prior.numel() == search_size:
        return target_prior.reshape(-1)
    if search_latent_shape and target_prior.numel() == search_latent_shape[-1]:
        view_shape = (1,) * (len(search_latent_shape) - 1) + (-1,)
        return target_prior.reshape(view_shape).expand(search_latent_shape).reshape(-1)
    raise ValueError(
        "Class target prior is incompatible with the counterfactual search latent shape."
    )


def optimize_z_counterfactuals(
    z0,
    w0,
    decoder_black_box_clf,
    total_CFs,
    min_iter,
    max_iter,
    proximity_weight_latent,
    proximity_weight_input,
    diversity_weight,
    latent_init_radius,
    target_class_index,
    original_input_instance,
    immutable_mask,
    monotone_increase_mask,
    monotone_decrease_mask,
    proximity_input_loss,
    continuous_indexes,
    categorical_slices,
    num_feature_weights,
    validity_loss,
    validity_margin,
    diversity_loss_type,
    plausibility_weight_latent,
    plausibility_latent_loss,
    plausibility_model,
    device,
    min_valid_to_stop=None,
    enforce_constraints=True,
    search_component="z",
    latent_init_center=None,
    trajectory_quality_selection=False,
    trajectory_snapshot_interval=100,
):
    if search_component not in {"z", "w"}:
        raise ValueError(f"Unknown search component: {search_component}")
    latent_center = z0 if search_component == "z" else w0
    min_valid_to_stop = min_valid_to_stop or total_CFs
    candidate_center = latent_center if latent_init_center is None else latent_init_center
    latent_candidates = initialize_latent_candidates(
        candidate_center, total_CFs, latent_init_radius, device
    )
    latent_candidates.requires_grad_(True)
    for param in decoder_black_box_clf.parameters():
        param.requires_grad_(False)
    if search_component == "z":
        decoder_black_box_clf.set_fixed_w(w0)
        decode_candidates = decoder_black_box_clf.decode_z
    else:
        decoder_black_box_clf.set_fixed_z(z0)
        decode_candidates = decoder_black_box_clf.decode_w
    if enforce_constraints:
        decoder_black_box_clf.set_actionability_reference(
            immutable_mask,
            monotone_increase_mask,
            monotone_decrease_mask,
            original_input_instance,
        )
    else:
        decoder_black_box_clf.clear_actionability_reference()
    optimizer = torch.optim.Adam([latent_candidates], lr=0.05)
    target = torch.full(
        (total_CFs,), target_class_index, dtype=torch.long, device=device
    )
    original_input_instance = torch.tensor(
        original_input_instance, dtype=torch.float32, device=device
    ).repeat(total_CFs, 1)

    best_valid = None
    best_score = None
    best_decoded = None
    trajectory_snapshots = []

    for step in range(max_iter):
        optimizer.zero_grad()
        decoded = decode_candidates(latent_candidates)
        logits = decoder_black_box_clf.bb_only(decoded)
        validity_loss_value = compute_validity_loss(
            logits, target, target_class_index, validity_loss, validity_margin
        )
        latent_prox = (
            (latent_candidates - latent_center.unsqueeze(0)) ** 2
        ).mean()
        if proximity_input_loss == "reconstruction":
            input_prox = compute_tabcf_input_proximity(
                decoded,
                original_input_instance,
                continuous_indexes,
                categorical_slices,
                num_feature_weights,
            )
        else:
            input_prox = ((decoded - original_input_instance) ** 2).mean()
        if total_CFs > 1:
            if diversity_loss_type in {"dpp", "dpp_det"}:
                diversity_loss = compute_dpp_diversity(decoded, variant="det")
            elif diversity_loss_type == "dpp_log":
                diversity_loss = compute_dpp_diversity(decoded, variant="log")
            else:
                diversity_loss = -torch.pdist(latent_candidates, p=2).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=device)
        if plausibility_weight_latent > 0:
            if plausibility_latent_loss != "GMM":
                raise ValueError(
                    f"Unknown plausibility_latent_loss: {plausibility_latent_loss}"
                )
            plausibility_loss = compute_latent_gmm_plausibility_loss(
                latent_candidates, plausibility_model
            )
        else:
            plausibility_loss = torch.tensor(0.0, device=device)
        loss = (
            validity_loss_value
            + proximity_weight_latent * latent_prox
            + proximity_weight_input * input_prox
            + diversity_weight * diversity_loss
            + plausibility_weight_latent * plausibility_loss
        )
        loss.backward()

        should_stop = False
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)[:, target_class_index]
            mask = probs > 0.5
            if mask.any():
                valid_count = int(mask.sum().item())
                score = valid_count + probs[mask].mean().item()
                if best_score is None or score > best_score:
                    best_score = score
                    best_valid = latent_candidates[mask].detach().cpu().clone()
                    best_decoded = decoded[mask].detach().cpu()
                if (
                    trajectory_quality_selection
                    and step >= min_iter
                    and (step - min_iter) % trajectory_snapshot_interval == 0
                    and valid_count >= min_valid_to_stop
                ):
                    trajectory_snapshots.append(
                        {
                            "step": step,
                            "latents": latent_candidates[mask].detach().cpu().clone(),
                            "decoded": decoded[mask].detach().cpu().clone(),
                            "target_probs": probs[mask].detach().cpu().clone(),
                        }
                    )
            should_stop = (
                not trajectory_quality_selection
                and step >= min_iter
                and best_valid is not None
                and len(best_valid) >= min_valid_to_stop
            )
        optimizer.step()
        if should_stop:
            break

    if best_valid is None:
        if trajectory_quality_selection:
            return None, None, []
        return None, None
    if trajectory_quality_selection:
        return best_valid, best_decoded, trajectory_snapshots
    return best_valid, best_decoded


def optimize_w_counterfactuals(w0, z0, *args, **kwargs):
    kwargs["search_component"] = "w"
    return optimize_z_counterfactuals(z0, w0, *args, **kwargs)


def main(args):

    dataname = args.dataname
    data_dir = f"data/{dataname}"

    device = args.device
    num_samples = args.num_samples
    source_start = getattr(args, "source_start", 0)
    save_path = args.save_path
    proximity_weight_input = args.proximity_weight_input
    proximity_weight_latent = args.proximity_weight_latent
    enforce_constraints = getattr(args, "enforce_constraints", False)
    trajectory_quality_selection = getattr(
        args, "trajectory_quality_selection", False
    )

    optimization_parameters = get_optim_params_form_args(args)

    dice_method = args.dice_method

    total_CFs = args.total_CFs
    candidate_CFs = getattr(args, "candidate_CFs", 0) or total_CFs
    candidate_CFs = max(candidate_CFs, total_CFs)
    target_class_index = args.target_class
    min_iter = args.min_iter
    max_iter = args.max_iter

    output_file_name_csvae = get_name_form_args(args)

    output_folder_parts = [
        dataname,
        "MDPCF",
        output_file_name_csvae,
        dice_method,
        optimization_parameters,
    ]
    sample_tag = build_sample_tag(num_samples, total_CFs, source_start)
    output_folder_parts.append(sample_tag)

    save_path = os.path.join(save_path, *output_folder_parts)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if os.listdir(save_path) and not is_resumable_sampling_directory(save_path):
            print("CFs for this setting already computed")
            exit(0)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    save_run_config(args, save_path)

    generation_inputs = get_input_generate(args)
    train_z = generation_inputs["train_z"]
    train_w = generation_inputs["train_w"]
    test_z = generation_inputs["test_z"]
    test_w = generation_inputs["test_w"]
    w_prior_mu = generation_inputs.get("w_prior_mu")
    info = generation_inputs["info"]
    num_inverse = generation_inputs["num_inverse"]
    cat_inverse = generation_inputs["cat_inverse"]
    attach_categorical_value_orders(info, cat_inverse)
    test_z_flat = test_z.reshape(test_z.shape[0], -1)
    test_w_flat = test_w.reshape(test_w.shape[0], -1)
    latent_model = getattr(args, "latent_model", "csvae")
    search_component = "w" if latent_model == "csvae" else "z"
    search_train_latents = train_w if search_component == "w" else train_z
    search_test_flat = test_w_flat if search_component == "w" else test_z_flat
    latent_prefix = search_component
    target_prior_init_weight = getattr(args, "target_prior_init_weight", 0.0)
    target_w_prior = None
    if search_component == "w" and target_prior_init_weight > 0:
        if w_prior_mu is None:
            raise RuntimeError(
                "Target-prior initialization requires w_prior_mu.npy artifacts."
            )
        target_prior_token = w_prior_mu[target_class_index].to(device)
        target_w_prior = expand_target_prior_for_search(
            target_prior_token, search_train_latents.shape[1:]
        )

    test_set_path = f"data/{dataname}/test.csv"

    test_df = pd.read_csv(test_set_path)

    column_names = info["column_names"].copy()
    target_col = info["target_col"]
    column_names.remove(target_col)

    test_z_df = pd.DataFrame(test_z_flat)
    test_w_df = pd.DataFrame(
        test_w_flat, columns=[f"w_{i}" for i in range(test_w_flat.shape[1])]
    )

    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir,
        task_type=info["task_type"],
        cat_encoding="one-hot",
        num_encoding="minmax",
    )
    continuous_indexes, categorical_slices = build_encoded_feature_groups(info)
    num_feature_weights = compute_num_feature_weights(
        X_num_cat_ohe[0], continuous_indexes
    )
    """
        Generating samples    
    """
    start_time = time.time()

    model_filename = get_black_box_model_path(args, data_dir)

    input_shape = X_num_cat_ohe[0].shape[-1]

    loaded_black_box_clf = BBMLPCLF(input_shape).to(device)

    loaded_black_box_clf.load_state_dict(torch.load(model_filename, map_location=device))
    # Ensure black_box_clf parameters are not updated
    for param in loaded_black_box_clf.parameters():
        param.requires_grad = False

    loaded_black_box_clf.eval()

    pre_decoder = info["pre_decoder"].to(device)
    pre_decoder.eval()

    test_tensor_input = torch.tensor(
        X_num_cat_ohe[1], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        test_logits = loaded_black_box_clf(test_tensor_input)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)

    (
        immutable_mask_np,
        monotone_increase_mask_np,
        monotone_decrease_mask_np,
    ) = build_actionability_masks(info, X_num_cat_ohe[0].shape[1])

    decoder_black_box_clf = Decoder_Black_Box(
        loaded_black_box_clf,
        pre_decoder,
        info,
        categorical_straight_through=args.categorical_straight_through,
    ).to(device)

    decoder_black_box_clf.eval()

    print("Loaded black box model from", model_filename)

    plausibility_weight_latent = max(
        float(getattr(args, "plausibility_weight_latent", -1.0)), 0.0
    )
    plausibility_model = None
    if plausibility_weight_latent > 0:
        plausibility_model = fit_latent_gmm_plausibility(
            search_train_latents.detach().cpu().numpy(),
            n_components=getattr(args, "components", 5),
            max_samples=getattr(args, "samples", 1000),
            device=device,
        )

    indices_negative_class = select_source_indices(
        test_preds,
        target_class_index=target_class_index,
        num_samples=num_samples,
        source_start=source_start,
    )
    source_manifest = pd.DataFrame(
        {
            "selected_order": np.arange(len(indices_negative_class), dtype=int),
            "test_index": indices_negative_class.astype(int),
            "black_box_pred": test_preds[indices_negative_class].astype(int),
            "target_prob": test_probs[indices_negative_class, target_class_index],
        }
    )
    source_manifest.to_csv(
        os.path.join(save_path, "source_manifest.csv"), index=False
    )

    negative_test_df = test_df.iloc[indices_negative_class].reset_index(drop=True)

    negative_test_z_df = test_z_df.iloc[indices_negative_class].reset_index(drop=True)
    negative_test_w_df = test_w_df.iloc[indices_negative_class].reset_index(drop=True)

    negative_test_encoded = X_num_cat_ohe[1][indices_negative_class]
    immutable_ref_np = negative_test_encoded.copy()
    diversity_weight = getattr(args, "diversity_weight", 1.0)
    latent_init_radius = resolve_latent_init_radius(args, search_train_latents)
    cf_selection_metric = getattr(args, "cf_selection_metric", "mixed_distance")
    if trajectory_quality_selection and cf_selection_metric != "lof_dpp":
        raise ValueError(
            "trajectory_quality_selection requires cf_selection_metric=lof_dpp"
        )
    lof_selector_model = None
    if cf_selection_metric.startswith("lof_") or trajectory_quality_selection:
        lof_selector_model = fit_encoded_lof_selector(X_num_cat_ohe[0])

    print(
        {
            "total_CFs": total_CFs,
            "candidate_CFs": candidate_CFs,
            "cf_selection_metric": cf_selection_metric,
            "min_iter": min_iter,
            "max_iter": max_iter,
            "proximity_weight_input": proximity_weight_input,
            "proximity_weight_latent": proximity_weight_latent,
            "diversity_weight": diversity_weight,
            "diversity_loss": args.diversity_loss,
            "plausibility_weight_latent": plausibility_weight_latent,
            "plausibility_latent_loss": getattr(args, "plausibility_latent_loss", "GMM"),
            "latent_init_radius_strategy": getattr(
                args, "latent_init_radius_strategy", "fixed"
            ),
            "latent_init_radius": latent_init_radius,
            "search_component": search_component,
            "target_prior_init_weight": target_prior_init_weight,
            "latent_init_radius_scale": getattr(args, "latent_init_radius_scale", 1.0),
            "enforce_constraints": enforce_constraints,
        }
    )

    checkpoint_root = os.path.join(save_path, "checkpoints")
    completed_sources = {
        source_id: marker
        for source_id, marker in load_completed_sources(checkpoint_root).items()
        if 0 <= source_id < len(negative_test_z_df)
    }
    sampling_stats = {
        "attempted_count": len(negative_test_z_df),
        "optimized_valid_source_count": 0,
        "optimized_valid_candidate_count": 0,
        "post_projection_candidate_count": 0,
        "post_projection_validity_count": 0,
        "post_projection_constraints_count": 0,
        "saved_cf_count": 0,
        "saved_source_count": 0,
    }
    completed_stats = aggregate_sampling_stats(completed_sources)
    for key in sampling_stats:
        if key != "attempted_count":
            sampling_stats[key] = completed_stats.get(key, 0)
    print(
        f"Resuming sampling: {len(completed_sources)}/{len(negative_test_z_df)} "
        "source checkpoints complete."
    )

    for indx, row in tqdm(
        negative_test_z_df.iterrows(),
        total=len(negative_test_z_df),
        desc="CFs for test set",
    ):
        if indx in completed_sources:
            continue
        stats_before_source = sampling_stats.copy()
        source_shards = {}
        z0 = torch.tensor(row.to_numpy(), dtype=torch.float32, device=device)
        w0 = torch.tensor(
            negative_test_w_df.iloc[indx].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        w0 = resolve_fixed_decoder_context(
            latent_model, w0, target_class_index, device
        )
        optimize_counterfactuals = (
            optimize_w_counterfactuals
            if search_component == "w"
            else optimize_z_counterfactuals
        )
        source_latent_args = (
            (w0, z0) if search_component == "w" else (z0, w0)
        )
        optimization_result = optimize_counterfactuals(
            *source_latent_args,
            decoder_black_box_clf,
            candidate_CFs,
            min_iter,
            max_iter,
            max(proximity_weight_latent, 0.0),
            max(proximity_weight_input, 0.0),
            diversity_weight,
            latent_init_radius,
            target_class_index,
            immutable_ref_np[indx : indx + 1],
            torch.tensor(immutable_mask_np, dtype=torch.bool, device=device),
            torch.tensor(monotone_increase_mask_np, dtype=torch.bool, device=device),
            torch.tensor(monotone_decrease_mask_np, dtype=torch.bool, device=device),
            args.proximity_input_loss,
            continuous_indexes,
            categorical_slices,
            num_feature_weights,
            args.validity_loss,
            args.validity_margin,
            args.diversity_loss,
            plausibility_weight_latent,
            args.plausibility_latent_loss,
            plausibility_model,
            device,
            min_valid_to_stop=total_CFs,
            enforce_constraints=enforce_constraints,
            latent_init_center=(
                resolve_latent_initialization_center(
                    w0, target_w_prior, target_prior_init_weight
                )
                if search_component == "w"
                else None
            ),
            trajectory_quality_selection=trajectory_quality_selection,
            trajectory_snapshot_interval=100,
        )
        if trajectory_quality_selection:
            latent_valid, decoded_valid, trajectory_snapshots = optimization_result
            if trajectory_snapshots:
                original_row = negative_test_df.loc[indx]
                prepared_snapshots = prepare_trajectory_quality_snapshots(
                    trajectory_snapshots,
                    info,
                    num_inverse,
                    cat_inverse,
                    original_row,
                    loaded_black_box_clf,
                    lof_selector_model,
                    target_class_index,
                    total_CFs,
                    device,
                    enforce_constraints=enforce_constraints,
                )
                if prepared_snapshots:
                    selected_snapshot, trajectory_diagnostics = (
                        select_trajectory_quality_candidate_set(
                            prepared_snapshots, info, original_row, total_CFs
                        )
                    )
                    latent_valid = selected_snapshot["latents"]
                    decoded_valid = selected_snapshot["decoded"]
                    trajectory_frame = pd.DataFrame(trajectory_diagnostics)
                    trajectory_frame.insert(0, "id", indx)
                    source_shards["trajectory_quality"] = trajectory_frame
                else:
                    latent_valid, decoded_valid = None, None
            else:
                latent_valid, decoded_valid = None, None
        else:
            latent_valid, decoded_valid = optimization_result
        if latent_valid is None:
            commit_source_sampling_checkpoint(
                checkpoint_root,
                indx,
                len(negative_test_z_df),
                stats_before_source,
                sampling_stats,
                source_shards,
            )
            continue
        sampling_stats["optimized_valid_source_count"] += 1
        sampling_stats["optimized_valid_candidate_count"] += len(latent_valid)
        cfs = recover_data_from_encoded(
            decoded_valid.numpy(), info, num_inverse, cat_inverse
        )
        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        cfs.rename(columns=idx_name_mapping, inplace=True)
        original_row = negative_test_df.loc[indx]
        if enforce_constraints:
            apply_raw_actionability_constraints(cfs, original_row, info)
            apply_numerical_domain_constraints(cfs, info)
            apply_integer_constraints(cfs, info)
            apply_raw_actionability_constraints(cfs, original_row, info)
            apply_numerical_domain_constraints(cfs, info)
        rounded_encoded = encode_raw_features_for_black_box(cfs, info)
        with torch.no_grad():
            before_probs = torch.softmax(
                decoder_black_box_clf.bb_only(decoded_valid.to(device)),
                dim=-1,
            )[:, target_class_index]
            probs = torch.softmax(
                decoder_black_box_clf.bb_only(
                    torch.tensor(rounded_encoded, dtype=torch.float32, device=device)
                ),
                dim=-1,
            )[:, target_class_index]
        sampling_stats["post_projection_candidate_count"] += len(cfs)
        validity_mask = _as_numpy(probs) > 0.5
        constraints_mask = (
            validate_counterfactual_constraints(cfs, original_row, info)
            if enforce_constraints
            else np.ones(len(cfs), dtype=bool)
        )
        valid_after_rounding = validity_mask & constraints_mask
        source_shards["projection"] = build_projection_shift_summary(
            source_id=indx,
            decoded_encoded=decoded_valid,
            rounded_encoded=rounded_encoded,
            before_probs=before_probs,
            after_probs=probs,
            validity_mask=validity_mask,
            constraints_mask=constraints_mask,
            info=info,
        )
        sampling_stats["post_projection_validity_count"] += int(validity_mask.sum())
        sampling_stats["post_projection_constraints_count"] += int(
            constraints_mask.sum()
        )
        if not valid_after_rounding.any():
            commit_source_sampling_checkpoint(
                checkpoint_root,
                indx,
                len(negative_test_z_df),
                stats_before_source,
                sampling_stats,
                source_shards,
            )
            continue
        cfs = cfs.loc[valid_after_rounding].reset_index(drop=True)
        cfs_latents = latent_valid[valid_after_rounding].reshape(
            int(valid_after_rounding.sum()), -1
        ).numpy()
        target_probs_final = probs.detach().cpu().numpy()
        target_probs_final = target_probs_final[valid_after_rounding]

        def predict_target_prob(raw_df):
            encoded = encode_raw_features_for_black_box(raw_df, info)
            with torch.no_grad():
                logits = loaded_black_box_clf(
                    torch.tensor(encoded, dtype=torch.float32, device=device)
                )
                return torch.softmax(logits, dim=-1)[:, target_class_index].cpu().numpy()

        if getattr(args, "posthoc_sparsity", False):
            cfs_before_posthoc = cfs.copy()
            cfs_before_posthoc["id"] = indx
            cfs_before_posthoc["target_prob"] = target_probs_final
            source_shards["cfs_before_posthoc_sparsity"] = cfs_before_posthoc

            cfs, target_probs_final, sparsity_summary = apply_dice_posthoc_sparsity(
                cfs,
                original_row,
                info,
                predict_target_prob,
                threshold=0.5,
                enforce_constraints=enforce_constraints,
            )
            sparsity_summary.insert(0, "id", indx)
            source_shards["posthoc_sparsity"] = sparsity_summary
        cfs_latents_df = pd.DataFrame(
            cfs_latents,
            columns=[f"{latent_prefix}_{i}" for i in range(cfs_latents.shape[1])],
        )
        candidate_lof_scores = (
            score_raw_candidates_with_lof(cfs, info, lof_selector_model)
            if cf_selection_metric.startswith("lof_")
            else None
        )
        cfs, cfs_latents_df, target_probs_final = select_diverse_counterfactual_subset(
            cfs,
            cfs_latents_df,
            target_probs_final,
            info,
            total_CFs,
            selection_metric=cf_selection_metric,
            candidate_plausibility_scores=candidate_lof_scores,
        )
        cfs = cfs.reset_index(drop=True)
        cfs_latents = cfs_latents_df.reset_index(drop=True).to_numpy()
        if getattr(args, "posthoc_diversity", False):
            cfs, target_probs_final, diversity_summary = apply_posthoc_categorical_diversity(
                cfs,
                original_row,
                info,
                predict_target_prob,
                threshold=0.5,
                enforce_constraints=enforce_constraints,
            )
            diversity_summary.insert(0, "id", indx)
            source_shards["posthoc_diversity"] = diversity_summary
        final_valid_mask = validate_counterfactuals(
            cfs,
            original_row,
            info,
            target_probs_final,
            threshold=0.5,
            enforce_constraints=enforce_constraints,
        )
        if not final_valid_mask.any():
            commit_source_sampling_checkpoint(
                checkpoint_root,
                indx,
                len(negative_test_z_df),
                stats_before_source,
                sampling_stats,
                source_shards,
            )
            continue
        cfs = cfs.loc[final_valid_mask].reset_index(drop=True)
        cfs_latents = cfs_latents[final_valid_mask]
        target_probs_final = target_probs_final[final_valid_mask]
        cfs["id"] = indx
        cfs["target_prob"] = target_probs_final
        latent_df = pd.DataFrame(
            cfs_latents,
            columns=[f"{latent_prefix}_{i}" for i in range(cfs_latents.shape[1])],
        )
        latent_df.insert(0, "id", indx)
        sampling_stats["saved_cf_count"] += len(cfs)
        sampling_stats["saved_source_count"] += 1
        source_shards["cfs"] = cfs
        source_shards["latents"] = latent_df
        commit_source_sampling_checkpoint(
            checkpoint_root,
            indx,
            len(negative_test_z_df),
            stats_before_source,
            sampling_stats,
            source_shards,
        )

    cols = ["id"] + column_names
    syn_df = merge_source_shards(
        checkpoint_root, "cfs", columns=cols + ["target_prob"]
    )
    cf_latents_df = merge_source_shards(
        checkpoint_root,
        "latents",
        columns=["id"]
        + [f"{latent_prefix}_{i}" for i in range(search_test_flat.shape[1])],
    )
    if syn_df.empty:
        print("No valid counterfactuals generated for current settings.")

    syn_df.to_csv(os.path.join(save_path, "cfs.csv"), index=False)
    cf_latents_df.to_csv(os.path.join(save_path, "cf_latents.csv"), index=False)

    if getattr(args, "posthoc_sparsity", False):
        before_posthoc_df = merge_source_shards(
            checkpoint_root,
            "cfs_before_posthoc_sparsity",
            columns=cols + ["target_prob"],
        )
        before_posthoc_df.to_csv(
            os.path.join(save_path, "cfs_before_posthoc_sparsity.csv"), index=False
        )

    end_time = time.time()
    print("Time:", end_time - start_time)

    negative_test_df["id"] = list(range(len(negative_test_df)))
    negative_test_df = negative_test_df[cols]

    negative_test_df.to_csv(
        os.path.join(save_path, "original_test_samples.csv"), index=False
    )

    feature_change_summary = build_feature_change_summary(syn_df, negative_test_df, info)
    feature_change_summary.to_csv(
        os.path.join(save_path, "feature_change_summary.csv"), index=False
    )
    sampling_stats = aggregate_sampling_stats(load_completed_sources(checkpoint_root))
    sampling_stats["attempted_count"] = len(negative_test_z_df)
    sampling_diagnostics = build_sampling_diagnostics(sampling_stats)
    sampling_diagnostics.to_csv(
        os.path.join(save_path, "sampling_diagnostics.csv"), index=False
    )
    projection_shift_summary = merge_source_shards(
        checkpoint_root, "projection", columns=None
    )
    projection_shift_summary.to_csv(
        os.path.join(save_path, "projection_shift_summary.csv"), index=False
    )
    if trajectory_quality_selection:
        trajectory_quality_summary = merge_source_shards(
            checkpoint_root,
            "trajectory_quality",
            columns=[
                "id",
                "step",
                "diversity",
                "lof",
                "target_prob",
                "proximity",
                "quality_score",
                "selected",
            ],
        )
        trajectory_quality_summary.to_csv(
            os.path.join(save_path, "trajectory_quality_summary.csv"), index=False
        )
    posthoc_sparsity_summary = merge_source_shards(
        checkpoint_root,
        "posthoc_sparsity",
        columns=[
            "id",
            "cf_index",
            "target_prob_before_sparsity",
            "target_prob_after_sparsity",
            "continuous_revert_attempted",
            "continuous_revert_accepted",
            "changed_features_before",
            "changed_features_after",
        ],
    )
    posthoc_sparsity_summary.to_csv(
        os.path.join(save_path, "posthoc_sparsity_summary.csv"), index=False
    )
    posthoc_diversity_summary = merge_source_shards(
        checkpoint_root,
        "posthoc_diversity",
        columns=[
            "id",
            "candidate_index",
            "categorical_diversity_attempted",
            "categorical_diversity_accepted",
            "target_prob_after_diversity",
        ],
    )
    posthoc_diversity_summary.to_csv(
        os.path.join(save_path, "posthoc_diversity_summary.csv"), index=False
    )
    print(sampling_diagnostics.to_dict(orient="records")[0])

    print("Saving sampled data to {}".format(save_path))
