from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    top_p_percent: float,
    alpha: float,
    beta: float,
    gamma: float,
    stage_debug: dict,
    decoded_summary: dict,
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

    # Visualization for before/after selection in the same style as stage1 (PCA + cluster centers).
    x_for_plot = np.column_stack(
        [
            np.asarray(novelty_n, dtype=np.float64),
            np.asarray(coverage_n, dtype=np.float64),
            np.asarray(redundancy_n, dtype=np.float64),
            np.asarray(learnability, dtype=np.float64),
        ]
    )
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_for_plot)
    pca = PCA(n_components=2, random_state=0)
    x2d = pca.fit_transform(xs)

    uniq_clusters = np.unique(cluster_ids)
    centers = []
    for cid in uniq_clusters.tolist():
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        centers.append(xs[idx].mean(axis=0))
    centers = np.asarray(centers, dtype=np.float64)
    c2d = pca.transform(centers) if centers.size > 0 else np.empty((0, 2), dtype=np.float64)

    sel_idx = np.where(in_pool == 1)[0]
    rej_idx = np.where(in_pool == 0)[0]
    vis_png = output_dir / "stage2_selection_visualization.png"
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(x2d[:, 0], x2d[:, 1], c=cluster_ids, cmap="tab10", s=12, alpha=0.7, edgecolors="none")
    if c2d.shape[0] > 0:
        ax1.scatter(c2d[:, 0], c2d[:, 1], c="red", s=90, marker="o", edgecolors="none", zorder=4, label="Cluster Centers")
        ax1.legend(loc="best")
    ax1.set_title("Stage2 Before Selection")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    fig.colorbar(sc1, ax=ax1, label="cluster_id")

    ax2 = fig.add_subplot(1, 2, 2)
    if rej_idx.size > 0:
        ax2.scatter(x2d[rej_idx, 0], x2d[rej_idx, 1], c="#b6b6b6", s=10, alpha=0.35, edgecolors="none", label="Not Selected")
    if sel_idx.size > 0:
        sc2 = ax2.scatter(x2d[sel_idx, 0], x2d[sel_idx, 1], c=cluster_ids[sel_idx], cmap="tab10", s=18, alpha=0.9, edgecolors="none", label="Selected")
        fig.colorbar(sc2, ax=ax2, label="selected cluster_id")
    if c2d.shape[0] > 0:
        ax2.scatter(c2d[:, 0], c2d[:, 1], c="red", s=90, marker="o", edgecolors="none", zorder=4, label="Cluster Centers")
    ax2.set_title("Stage2 After Selection")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(vis_png, dpi=220)
    plt.close(fig)

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
        "top_p_percent": float(top_p_percent),
        "representative_pool_size": int(len(rep_indices)),
        "weights": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        },
        "debug": stage_debug,
        "decoded_summary": decoded_summary,
        "output_files": {
            "representative_pool": str(rep_csv),
            "learnability_scores": str(score_csv),
            "learnability_npy": str(learnability_npy),
            "representative_pool_npy": str(pool_npy),
            "learnability_pt": str(learnability_pt),
            "representative_pool_pt": str(pool_pt),
            "selection_visualization": str(vis_png),
        },
    }

    summary_path = output_dir / "learnability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "summary_json": str(summary_path),
        "scores_csv": str(score_csv),
        "representative_pool_size": int(len(rep_indices)),
    }
