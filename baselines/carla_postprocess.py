import numpy as np


def get_encoded_category_groups(dataset):
    if hasattr(dataset.encoder, "get_feature_names"):
        encoded_feature_names = list(dataset.encoder.get_feature_names(dataset.categorical))
    else:
        encoded_feature_names = list(dataset.encoder.get_feature_names_out(dataset.categorical))

    groups = []
    start = 0
    for categories in dataset.encoder.categories_:
        end = start + len(categories)
        groups.append(encoded_feature_names[start:end])
        start = end
    return groups


def project_encoded_categories(df, dataset):
    projected_df = df.copy()

    for group in get_encoded_category_groups(dataset):
        present_cols = [col for col in group if col in projected_df.columns]
        if not present_cols:
            continue

        values = projected_df[present_cols].to_numpy(dtype=float)
        selected = np.argmax(values, axis=1)
        projected = np.zeros_like(values, dtype=float)
        projected[np.arange(values.shape[0]), selected] = 1.0
        projected_df.loc[:, present_cols] = projected

    return projected_df
