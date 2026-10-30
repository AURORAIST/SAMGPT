from __future__ import annotations

import numpy as np

from models.stage1_gmm_cluster import fit_clusters as fit_clusters_gmm
from models.stage1_kmeans_cluster import fit_clusters as fit_clusters_kmeans


def fit_clusters(
    records: list[dict[str, float | int]],
    num_clusters: int,
    seed: int,
    use_log1p: bool,
    encoding_mode: str,
    cluster_method: str = "gmm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    method = str(cluster_method).strip().lower()
    if method == "gmm":
        return fit_clusters_gmm(
            records=records,
            num_clusters=num_clusters,
            seed=seed,
            use_log1p=use_log1p,
            encoding_mode=encoding_mode,
        )

    if method == "kmeans":
        return fit_clusters_kmeans(
            records=records,
            num_clusters=num_clusters,
            seed=seed,
            use_log1p=use_log1p,
            encoding_mode=encoding_mode,
        )

    raise ValueError(f"Unsupported cluster_method={cluster_method}. Use 'gmm' or 'kmeans'.")
