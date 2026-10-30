from __future__ import annotations

import numpy as np

FEATURE_KEYS = [
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


def _minmax(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return x / norm


def compute_learnability(
    x: np.ndarray,
    cluster_ids: np.ndarray,
    rep_indices: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xn = _row_l2_normalize(x.astype(np.float32))
    sim = xn @ xn.T

    if rep_indices.size > 0:
        baseline_coverage = sim[:, rep_indices].max(axis=1)
    else:
        baseline_coverage = np.zeros(sim.shape[0], dtype=np.float32)

    novelty = np.maximum(sim - baseline_coverage[None, :], 0.0).sum(axis=1)

    uniq, counts = np.unique(cluster_ids, return_counts=True)
    p_map = {int(c): float(cnt / len(cluster_ids)) for c, cnt in zip(uniq, counts)}
    coverage = np.array([-np.log(max(p_map[int(c)], 1e-12)) for c in cluster_ids], dtype=np.float64)

    redundancy = -baseline_coverage.astype(np.float64)

    novelty_n = _minmax(novelty.astype(np.float64))
    coverage_n = _minmax(coverage)
    redundancy_n = _minmax(redundancy)

    learnability = alpha * novelty_n + beta * coverage_n + gamma * redundancy_n
    order = np.argsort(-learnability)

    return novelty, coverage, redundancy, novelty_n, coverage_n, redundancy_n, learnability, order
