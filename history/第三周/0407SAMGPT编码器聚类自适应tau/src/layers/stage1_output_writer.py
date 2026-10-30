from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

LOG1P_FEATURE_NAMES = [
    "num_nodes",
    "num_edges",
    "max_degree",
    "var_degree",
    "triangles",
    "diameter",
]


def _save_cluster_visualization(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    output_path: Path,
    dataset_name: str,
    remove_selfloops: bool,
) -> None:
    pca = PCA(n_components=2, random_state=0)
    x2d = pca.fit_transform(x_scaled)
    c2d = pca.transform(centers_scaled)

    unique_clusters, counts = np.unique(labels, return_counts=True)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    scatter = ax1.scatter(
        x2d[:, 0],
        x2d[:, 1],
        c=labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    ax1.scatter(
        c2d[:, 0],
        c2d[:, 1],
        marker="o",
        s=90,
        c="red",
        edgecolors="none",
        zorder=4,
        label="Cluster Centers",
    )

    ax1.set_title("Stage-1 Cluster Visualization (PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label("cluster_id")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(unique_clusters, counts, color="#2f6c8f")
    ax2.set_title("Cluster Size Distribution")
    ax2.set_xlabel("cluster_id")
    ax2.set_ylabel("count")
    ax2.set_xticks(unique_clusters)

    fig.suptitle(
        f"{dataset_name} | remove_selfloops={int(remove_selfloops)} | clusters={len(unique_clusters)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_stage1_outputs(
    output_root: str,
    dataset: str,
    records: list[dict[str, float | int]],
    labels: np.ndarray,
    centers_raw: np.ndarray,
    x_scaled: np.ndarray,
    centers_scaled: np.ndarray,
    max_subgraph_nodes: int,
    subgraph_hop_n: int,
    num_clusters: int,
    remove_selfloops: bool,
    encoding_mode: str,
    cluster_method: str,
    use_log1p: bool,
    cluster_feature_names: list[str],
    feature_means: np.ndarray,
) -> dict[str, str | int]:
    output_dir = Path(output_root) / dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "structural_encodings.csv"
    npy_labels = output_dir / "cluster_labels.npy"
    npy_centers = output_dir / "cluster_centers.npy"
    viz_path = output_dir / "stage1_cluster_visualization.png"
    summary_json = output_dir / "stage1_summary.json"

    header = [
        "sample_id",
        "graph_id",
        "subgraph_id",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "max_degree",
        "var_degree",
        "density",
        "avg_clustering",
        "triangles",
        "diameter",
        "cluster_id",
    ]

    lines = [",".join(header)]
    for rec, cluster in zip(records, labels.tolist()):
        row = [
            str(rec["sample_id"]),
            str(rec["graph_id"]),
            str(rec["subgraph_id"]),
            str(rec["num_nodes"]),
            str(rec["num_edges"]),
            str(rec["avg_degree"]),
            str(rec["max_degree"]),
            str(rec["var_degree"]),
            str(rec["density"]),
            str(rec["avg_clustering"]),
            str(rec["triangles"]),
            str(rec["diameter"]),
            str(cluster),
        ]
        lines.append(",".join(row))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(npy_labels, labels)
    np.save(npy_centers, centers_raw)

    _save_cluster_visualization(
        x_scaled=x_scaled,
        labels=labels,
        centers_scaled=centers_scaled,
        output_path=viz_path,
        dataset_name=dataset,
        remove_selfloops=remove_selfloops,
    )

    total_samples = int(len(records))
    total_subgraphs = total_samples
    graph_ids = sorted({int(rec["graph_id"]) for rec in records})
    cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
    cluster_count_map = {
        str(int(cid)): int(cnt)
        for cid, cnt in zip(cluster_ids.tolist(), cluster_counts.tolist())
    }

    summary = {
        "dataset": dataset,
        "num_samples": total_samples,
        "total_samples": total_samples,
        "total_subgraphs": total_subgraphs,
        "total_graphs": int(len(graph_ids)),
        "graph_ids": graph_ids,
        "cluster_sample_count_map": cluster_count_map,
        "max_subgraph_nodes": int(max_subgraph_nodes),
        "subgraph_hop_n": int(subgraph_hop_n),
        "cluster_method": str(cluster_method),
        "encoding_mode": encoding_mode,
        "cluster_feature_names": cluster_feature_names,
        "use_log1p": bool(use_log1p),
        "log1p_feature_names": LOG1P_FEATURE_NAMES,
        "requested_num_clusters": int(num_clusters),
        "effective_num_clusters": int(centers_raw.shape[0]),
        "remove_selfloops": bool(remove_selfloops),
        "feature_means": feature_means.tolist(),
        "paths": {
            "encodings_csv": str(csv_path),
            "cluster_labels": str(npy_labels),
            "cluster_centers": str(npy_centers),
            "cluster_visualization": str(viz_path),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "summary_json": str(summary_json),
        "effective_k": int(centers_raw.shape[0]),
    }
