from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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


@dataclass
class Stage2Config:
    input_csv: str
    output_dir: str
    top_m_per_cluster: int = 5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    center_path: str = ""


def _load_table(csv_path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    table: dict[str, np.ndarray] = {}
    for name in arr.dtype.names:
        table[name] = np.asarray(arr[name])
    return table


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


def _build_representative_pool(
    x: np.ndarray,
    cluster_ids: np.ndarray,
    sample_ids: np.ndarray,
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

    # Deduplicate while keeping first occurrence.
    if rep_indices_arr.size > 0:
        uniq, first_pos = np.unique(rep_indices_arr, return_index=True)
        keep = np.argsort(first_pos)
        rep_indices_arr = uniq[keep]
        rep_dist_arr = rep_dist_arr[first_pos[keep]]

    return rep_indices_arr, rep_dist_arr


def run_stage2(config: Stage2Config) -> dict[str, str | int | float]:
    csv_path = Path(config.input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"input csv not found: {csv_path}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = _load_table(csv_path)
    x = np.column_stack([table[k].astype(float) for k in FEATURE_KEYS]).astype(np.float64)
    sample_ids = table["sample_id"].astype(int)
    graph_ids = table["graph_id"].astype(int) if "graph_id" in table else np.zeros(len(sample_ids), dtype=np.int64)
    subgraph_ids = table["subgraph_id"].astype(int) if "subgraph_id" in table else np.arange(len(sample_ids), dtype=np.int64)
    cluster_ids = table["cluster_id"].astype(int)

    centers_raw = None
    if config.center_path:
        center_path = Path(config.center_path).resolve()
    else:
        modern_center = csv_path.parent / "cluster_centers.npy"
        legacy_center = csv_path.parent / "kmeans_centers.npy"
        center_path = modern_center if modern_center.exists() else legacy_center
    if center_path.exists():
        centers_raw = np.load(center_path)

    rep_indices, rep_dist = _build_representative_pool(
        x=x,
        cluster_ids=cluster_ids,
        sample_ids=sample_ids,
        top_m_per_cluster=config.top_m_per_cluster,
        centers_raw=centers_raw,
    )

    # Cosine similarity matrix in structural encoding space.
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

    learnability = (
        config.alpha * novelty_n
        + config.beta * coverage_n
        + config.gamma * redundancy_n
    )

    in_pool = np.zeros(len(sample_ids), dtype=np.int64)
    if rep_indices.size > 0:
        in_pool[rep_indices] = 1

    # representative pool
    rep_csv = output_dir / "representative_pool.csv"
    rep_lines = ["pool_id,sample_id,cluster_id,distance_to_center"]
    for i, idx in enumerate(rep_indices.tolist()):
        rep_lines.append(
            ",".join(
                [
                    str(i),
                    str(sample_ids[idx]),
                    str(cluster_ids[idx]),
                    str(float(rep_dist[i])) if i < len(rep_dist) else "0.0",
                ]
            )
        )
    rep_csv.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    score_csv = output_dir / "learnability_scores.csv"
    score_header = (
        "sample_id,graph_id,subgraph_id,cluster_id,in_representative_pool,novelty,coverage,redundancy,"
        "novelty_norm,coverage_norm,redundancy_norm,learnability"
    )
    score_rows = [score_header]

    order = np.argsort(-learnability)
    for idx in order.tolist():
        row = [
            str(int(sample_ids[idx])),
            str(int(graph_ids[idx])),
            str(int(subgraph_ids[idx])),
            str(int(cluster_ids[idx])),
            str(int(in_pool[idx])),
            str(float(novelty[idx])),
            str(float(coverage[idx])),
            str(float(redundancy[idx])),
            str(float(novelty_n[idx])),
            str(float(coverage_n[idx])),
            str(float(redundancy_n[idx])),
            str(float(learnability[idx])),
        ]
        score_rows.append(",".join(row))

    score_csv.write_text("\n".join(score_rows) + "\n", encoding="utf-8")

    # Binary outputs for direct training use.
    learnability_npy = output_dir / "learnability.npy"
    pool_npy = output_dir / "representative_pool.npy"
    learnability_pt = output_dir / "learnability_scores.pt"
    pool_pt = output_dir / "representative_pool.pt"

    np.save(learnability_npy, learnability)
    np.save(pool_npy, sample_ids[rep_indices] if rep_indices.size > 0 else np.array([], dtype=np.int64))

    torch.save(
        {
            "sample_id": torch.as_tensor(sample_ids, dtype=torch.long),
            "graph_id": torch.as_tensor(graph_ids, dtype=torch.long),
            "subgraph_id": torch.as_tensor(subgraph_ids, dtype=torch.long),
            "cluster_id": torch.as_tensor(cluster_ids, dtype=torch.long),
            "in_representative_pool": torch.as_tensor(in_pool, dtype=torch.long),
            "novelty": torch.as_tensor(novelty, dtype=torch.float32),
            "coverage": torch.as_tensor(coverage, dtype=torch.float32),
            "redundancy": torch.as_tensor(redundancy, dtype=torch.float32),
            "novelty_norm": torch.as_tensor(novelty_n, dtype=torch.float32),
            "coverage_norm": torch.as_tensor(coverage_n, dtype=torch.float32),
            "redundancy_norm": torch.as_tensor(redundancy_n, dtype=torch.float32),
            "learnability": torch.as_tensor(learnability, dtype=torch.float32),
            "rank_desc": torch.as_tensor(order, dtype=torch.long),
        },
        learnability_pt,
    )

    torch.save(
        {
            "pool_sample_id": torch.as_tensor(sample_ids[rep_indices], dtype=torch.long)
            if rep_indices.size > 0
            else torch.empty((0,), dtype=torch.long),
            "pool_graph_id": torch.as_tensor(graph_ids[rep_indices], dtype=torch.long)
            if rep_indices.size > 0
            else torch.empty((0,), dtype=torch.long),
            "pool_subgraph_id": torch.as_tensor(subgraph_ids[rep_indices], dtype=torch.long)
            if rep_indices.size > 0
            else torch.empty((0,), dtype=torch.long),
            "pool_cluster_id": torch.as_tensor(cluster_ids[rep_indices], dtype=torch.long)
            if rep_indices.size > 0
            else torch.empty((0,), dtype=torch.long),
            "pool_original_index": torch.as_tensor(rep_indices, dtype=torch.long),
            "distance_to_center": torch.as_tensor(rep_dist, dtype=torch.float32),
        },
        pool_pt,
    )

    summary = {
        "input_csv": str(csv_path),
        "num_samples": int(len(sample_ids)),
        "num_clusters": int(len(np.unique(cluster_ids))),
        "top_m_per_cluster": int(config.top_m_per_cluster),
        "representative_pool_size": int(len(rep_indices)),
        "weights": {
            "alpha": float(config.alpha),
            "beta": float(config.beta),
            "gamma": float(config.gamma),
        },
        "output_files": {
            "representative_pool": str(rep_csv),
            "learnability_scores": str(score_csv),
            "learnability_npy": str(learnability_npy),
            "representative_pool_npy": str(pool_npy),
            "learnability_pt": str(learnability_pt),
            "representative_pool_pt": str(pool_pt),
        },
    }

    summary_path = output_dir / "learnability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "num_samples": int(len(sample_ids)),
        "representative_pool_size": int(len(rep_indices)),
        "summary_json": str(summary_path),
        "scores_csv": str(score_csv),
    }
