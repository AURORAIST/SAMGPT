from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


STAGE1_FEATURE_KEYS = [
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


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def evaluate_stage1_artifacts(stage1_dir: str) -> dict:
    """Compute stage1 metrics and save visualizations under stage1_dir."""
    out_dir = Path(stage1_dir)
    csv_path = out_dir / "structural_encodings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"stage1 csv not found: {csv_path}")

    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    arr = np.atleast_1d(arr)
    if arr.size == 0:
        raise ValueError(f"stage1 csv is empty: {csv_path}")

    num_samples = int(arr.shape[0])
    graph_ids = np.asarray(arr["graph_id"], dtype=np.int64)
    cluster_ids = np.asarray(arr["cluster_id"], dtype=np.int64)

    unique_clusters, cluster_counts = np.unique(cluster_ids, return_counts=True)
    cluster_ratio = cluster_counts.astype(np.float64) / np.maximum(1, cluster_counts.sum())
    cluster_entropy = float(-np.sum(cluster_ratio * np.log(np.clip(cluster_ratio, 1e-12, 1.0))))

    feature_matrix = np.column_stack([np.asarray(arr[k], dtype=np.float64) for k in STAGE1_FEATURE_KEYS])
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(feature_matrix)

    # Clustering quality metrics.
    # These scores require at least 2 clusters and enough samples in clusters.
    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None
    if unique_clusters.size >= 2 and num_samples > unique_clusters.size:
        try:
            silhouette = _safe_float(silhouette_score(Xs, cluster_ids))
        except Exception:
            silhouette = None
        try:
            davies_bouldin = _safe_float(davies_bouldin_score(Xs, cluster_ids))
        except Exception:
            davies_bouldin = None
        try:
            calinski_harabasz = _safe_float(calinski_harabasz_score(Xs, cluster_ids))
        except Exception:
            calinski_harabasz = None

    metrics = {
        "num_samples": num_samples,
        "num_graphs": int(np.unique(graph_ids).size),
        "num_clusters": int(unique_clusters.size),
        "cluster_entropy": cluster_entropy,
        "largest_cluster_ratio": float(cluster_counts.max() / np.maximum(1, cluster_counts.sum())),
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
        "feature_mean": {k: _safe_float(v) for k, v in zip(STAGE1_FEATURE_KEYS, feature_mean.tolist())},
        "feature_std": {k: _safe_float(v) for k, v in zip(STAGE1_FEATURE_KEYS, feature_std.tolist())},
    }

    # Plot 1: cluster size bar
    fig = plt.figure(figsize=(8, 5))
    plt.bar(unique_clusters.astype(int), cluster_counts.astype(int), color="#2b6f8a")
    plt.title("Stage1 Cluster Size Distribution")
    plt.xlabel("cluster_id")
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_dir / "stage1_metrics_cluster_size.png", dpi=180)
    plt.close(fig)

    # Plot 2: PCA cluster scatter (distribution view)
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Xs)

    # Compute cluster centers in original feature space, then project with the same scaler+PCA.
    centers = []
    for cid in unique_clusters.tolist():
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        centers.append(feature_matrix[idx].mean(axis=0))
    centers = np.asarray(centers, dtype=np.float64)
    centers2 = pca.transform(scaler.transform(centers)) if centers.size > 0 else np.empty((0, 2))

    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        X2[:, 0],
        X2[:, 1],
        c=cluster_ids,
        cmap="tab10",
        s=8,
        alpha=0.7,
        edgecolors="none",
    )
    if centers2.shape[0] > 0:
        plt.scatter(
            centers2[:, 0],
            centers2[:, 1],
            c="red",
            s=90,
            marker="o",
            edgecolors="none",
            label="cluster centers",
            zorder=5,
        )
    plt.title("Stage1 Cluster PCA Distribution")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.colorbar(sc, label="cluster_id")
    if centers2.shape[0] > 0:
        plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "stage1_metrics_cluster_pca.png", dpi=180)
    plt.close(fig)

    # Plot 3: key feature distributions
    fig = plt.figure(figsize=(12, 8))
    for i, key in enumerate(["num_nodes", "num_edges", "density", "avg_clustering"]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.hist(np.asarray(arr[key], dtype=np.float64), bins=40, color="#4f9d69", alpha=0.85)
        ax.set_title(key)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_metrics_feature_hist.png", dpi=180)
    plt.close(fig)

    metrics_path = out_dir / "stage1_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "metrics_json": str(metrics_path),
        "cluster_size_png": str(out_dir / "stage1_metrics_cluster_size.png"),
        "cluster_pca_png": str(out_dir / "stage1_metrics_cluster_pca.png"),
        "feature_hist_png": str(out_dir / "stage1_metrics_feature_hist.png"),
    }


def evaluate_stage2_artifacts(stage2_dir: str) -> dict:
    """Compute stage2 metrics and save visualizations under stage2_dir."""
    out_dir = Path(stage2_dir)
    score_csv = out_dir / "learnability_scores.csv"
    if not score_csv.exists():
        raise FileNotFoundError(f"stage2 score csv not found: {score_csv}")

    arr = np.genfromtxt(score_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    arr = np.atleast_1d(arr)
    if arr.size == 0:
        raise ValueError(f"stage2 score csv is empty: {score_csv}")

    learnability = np.asarray(arr["learnability"], dtype=np.float64)
    in_pool = np.asarray(arr["in_representative_pool"], dtype=np.int64)
    novelty = np.asarray(arr["novelty"], dtype=np.float64)
    coverage = np.asarray(arr["coverage"], dtype=np.float64)
    redundancy = np.asarray(arr["redundancy"], dtype=np.float64)

    metrics = {
        "num_samples": int(arr.shape[0]),
        "representative_pool_size": int(in_pool.sum()),
        "representative_pool_ratio": float(in_pool.mean()),
        "learnability_mean": _safe_float(learnability.mean()),
        "learnability_std": _safe_float(learnability.std()),
        "learnability_p90": _safe_float(np.percentile(learnability, 90)),
        "novelty_mean": _safe_float(novelty.mean()),
        "coverage_mean": _safe_float(coverage.mean()),
        "redundancy_mean": _safe_float(redundancy.mean()),
    }

    # Plot 1: sorted learnability curve
    sorted_l = np.sort(learnability)[::-1]
    fig = plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(sorted_l) + 1), sorted_l, color="#c04f36", linewidth=1.8)
    plt.title("Stage2 Learnability Rank Curve")
    plt.xlabel("rank")
    plt.ylabel("learnability")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "stage2_metrics_learnability_curve.png", dpi=180)
    plt.close(fig)

    # Plot 2: component scatter
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(novelty, coverage, s=8, c=learnability, cmap="viridis", alpha=0.75)
    plt.colorbar(label="learnability")
    plt.title("Stage2 Novelty vs Coverage")
    plt.xlabel("novelty")
    plt.ylabel("coverage")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "stage2_metrics_novelty_coverage.png", dpi=180)
    plt.close(fig)

    metrics_path = out_dir / "stage2_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "metrics_json": str(metrics_path),
        "learnability_curve_png": str(out_dir / "stage2_metrics_learnability_curve.png"),
        "novelty_coverage_png": str(out_dir / "stage2_metrics_novelty_coverage.png"),
    }
