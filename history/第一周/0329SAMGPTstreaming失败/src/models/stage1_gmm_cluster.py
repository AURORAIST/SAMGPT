from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RAW_FEATURE_NAMES = [
    "num_nodes",
    "num_edges",
    "avg_degree",
    "max_degree",
    "var_degree",
    "density",
    "avg_clustering",
    "triangles",
    "diameter",
]

LOG1P_FEATURE_IDX = [0, 1, 3, 4, 7, 8]  # num_nodes,num_edges,max_degree,var_degree,triangles,diameter


def _matrix_from_records(records: Iterable[dict[str, float | int]]) -> np.ndarray:
    return np.array([[float(r[k]) for k in RAW_FEATURE_NAMES] for r in records], dtype=np.float64)


def _apply_log1p_selected(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for idx in LOG1P_FEATURE_IDX:
        out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
    return out


def _build_cluster_input(
    x_raw: np.ndarray,
    use_log1p: bool,
    encoding_mode: str,
) -> tuple[np.ndarray, list[str]]:
    mode = encoding_mode.strip().lower()
    if mode == "raw9":
        x_trans = _apply_log1p_selected(x_raw) if use_log1p else x_raw.copy()
        return x_trans, RAW_FEATURE_NAMES

    if mode != "decorrelated7":
        raise ValueError(f"Unsupported encoding_mode={encoding_mode}. Use 'raw9' or 'decorrelated7'.")

    num_nodes = np.maximum(x_raw[:, 0], 0.0)
    num_edges = np.maximum(x_raw[:, 1], 0.0)
    avg_degree = np.maximum(x_raw[:, 2], 0.0)
    max_degree = np.maximum(x_raw[:, 3], 0.0)
    var_degree = np.maximum(x_raw[:, 4], 0.0)
    density = np.clip(x_raw[:, 5], 0.0, 1.0)
    avg_clustering = np.clip(x_raw[:, 6], 0.0, 1.0)
    triangles = np.maximum(x_raw[:, 7], 0.0)
    diameter = np.maximum(x_raw[:, 8], 0.0)

    log_num_edges = np.log1p(num_edges)
    hub_ratio = max_degree / np.maximum(num_nodes - 1.0, 1.0)
    degree_cv = np.sqrt(var_degree) / np.maximum(avg_degree, 1e-6)
    tri_denom = np.maximum(num_nodes * np.maximum(num_nodes - 1.0, 0.0) * np.maximum(num_nodes - 2.0, 0.0) / 6.0, 1.0)
    triangle_rate = triangles / tri_denom
    diameter_norm = diameter / np.maximum(num_nodes - 1.0, 1.0)

    x_enc = np.column_stack([
        log_num_edges,
        hub_ratio,
        degree_cv,
        density,
        avg_clustering,
        triangle_rate,
        diameter_norm,
    ]).astype(np.float64)
    names = [
        "log_num_edges",
        "hub_ratio",
        "degree_cv",
        "density",
        "avg_clustering",
        "triangle_rate",
        "diameter_norm",
    ]
    return x_enc, names


def _cluster_centers_in_raw9(x_raw: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, x_raw.shape[1]), dtype=np.float64)
    global_mean = x_raw.mean(axis=0)
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            centers[cid] = global_mean
        else:
            centers[cid] = x_raw[idx].mean(axis=0)
    return centers


def fit_clusters(
    records: list[dict[str, float | int]],
    num_clusters: int,
    seed: int,
    use_log1p: bool,
    encoding_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not records:
        raise ValueError("No records to cluster.")

    x_raw = _matrix_from_records(records)
    x_cluster, cluster_feature_names = _build_cluster_input(
        x_raw=x_raw,
        use_log1p=use_log1p,
        encoding_mode=encoding_mode,
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_cluster)

    k = min(max(1, num_clusters), x_scaled.shape[0])
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, reg_covar=1e-6)
    labels = model.fit_predict(x_scaled)
    centers_scaled = model.means_

    centers_raw9 = _cluster_centers_in_raw9(x_raw=x_raw, labels=labels, k=k)
    return labels, centers_raw9, scaler.mean_, x_scaled, centers_scaled, cluster_feature_names
