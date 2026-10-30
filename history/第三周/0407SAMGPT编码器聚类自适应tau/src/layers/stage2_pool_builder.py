from __future__ import annotations

from pathlib import Path

import numpy as np


def load_table(csv_path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    table: dict[str, np.ndarray] = {}
    for name in arr.dtype.names:
        table[name] = np.asarray(arr[name])
    return table


def build_representative_pool(
    x: np.ndarray,
    cluster_ids: np.ndarray,
    top_m_per_cluster: int,
    centers_raw: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    rep_indices: list[int] = []
    rep_distances: list[float] = []

    uniq_clusters = np.unique(cluster_ids)
    for c in uniq_clusters:
        idx = np.where(cluster_ids == c)[0]
        if idx.size == 0:
            continue

        if centers_raw is not None and int(c) >= 0 and int(c) < centers_raw.shape[0]:
            center = centers_raw[int(c)]
        else:
            center = x[idx].mean(axis=0)

        d = np.linalg.norm(x[idx] - center[None, :], axis=1)
        order = np.argsort(d)
        take = min(top_m_per_cluster, len(order))
        chosen_local = order[:take]

        rep_indices.extend(idx[chosen_local].tolist())
        rep_distances.extend(d[chosen_local].tolist())

    rep_indices_arr = np.array(rep_indices, dtype=np.int64)
    rep_dist_arr = np.array(rep_distances, dtype=np.float64)

    if rep_indices_arr.size > 0:
        uniq, first_pos = np.unique(rep_indices_arr, return_index=True)
        keep = np.argsort(first_pos)
        rep_indices_arr = uniq[keep]
        rep_dist_arr = rep_dist_arr[first_pos[keep]]

    return rep_indices_arr, rep_dist_arr
