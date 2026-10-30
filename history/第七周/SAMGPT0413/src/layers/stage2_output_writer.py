from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def write_stage2_outputs(
    output_dir: Path,
    input_csv: Path,
    sample_ids: np.ndarray,
    graph_ids: np.ndarray,
    subgraph_ids: np.ndarray,
    cluster_ids: np.ndarray,
    rep_indices: np.ndarray,
    rep_dist: np.ndarray,
    in_pool: np.ndarray,
    novelty: np.ndarray,
    coverage: np.ndarray,
    redundancy: np.ndarray,
    novelty_n: np.ndarray,
    coverage_n: np.ndarray,
    redundancy_n: np.ndarray,
    learnability: np.ndarray,
    order: np.ndarray,
    top_m_per_cluster: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[str, str | int]:
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
        "input_csv": str(input_csv),
        "num_samples": int(len(sample_ids)),
        "num_clusters": int(len(np.unique(cluster_ids))),
        "top_m_per_cluster": int(top_m_per_cluster),
        "representative_pool_size": int(len(rep_indices)),
        "weights": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
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
        "summary_json": str(summary_path),
        "scores_csv": str(score_csv),
        "representative_pool_size": int(len(rep_indices)),
    }
