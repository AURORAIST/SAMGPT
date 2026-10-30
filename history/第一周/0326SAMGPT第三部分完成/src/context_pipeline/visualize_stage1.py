from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize stage1 structural encodings")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to structural_encodings.csv",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="",
        help="Output image path; defaults to <csv_dir>/stage1_visualization.png",
    )
    parser.add_argument(
        "--min-nodes-to-show",
        type=int,
        default=1,
        help="Only visualize subgraphs with num_nodes >= this value",
    )
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data: dict[str, np.ndarray] = {}
    for key in arr.dtype.names:
        data[key] = np.asarray(arr[key])
    return data


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.output_image).resolve() if args.output_image else csv_path.parent / "stage1_visualization.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_csv(csv_path)
    x = np.column_stack([data[k].astype(float) for k in FEATURE_KEYS])
    cluster_ids = data["cluster_id"].astype(int)
    num_nodes = data["num_nodes"].astype(float)

    # Keep small isolated subgraphs from dominating the visual pattern.
    mask = num_nodes >= float(args.min_nodes_to_show)
    x = x[mask]
    cluster_ids = cluster_ids[mask]

    if len(cluster_ids) == 0:
        raise ValueError("No samples left after filtering. Lower --min-nodes-to-show.")

    x_scaled = StandardScaler().fit_transform(x)

    # PCA in the same normalized space used by stage1 clustering.
    pca = PCA(n_components=2, random_state=0)
    x2d = pca.fit_transform(x_scaled)

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    sil = silhouette_score(x_scaled, cluster_ids) if len(unique_clusters) > 1 else float("nan")
    dbi = davies_bouldin_score(x_scaled, cluster_ids) if len(unique_clusters) > 1 else float("nan")

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    scatter = ax1.scatter(
        x2d[:, 0],
        x2d[:, 1],
        c=cluster_ids,
        cmap="tab10",
        s=24,
        alpha=0.85,
        edgecolors="none",
    )
    ax1.set_title("Stage-1 Structural Encodings (Standardized PCA 2D)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label("cluster_id")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(unique_clusters, counts, color="#2f6c8f")
    ax2.set_title("Cluster Size Distribution")
    ax2.set_xlabel("cluster_id")
    ax2.set_ylabel("count")
    ax2.set_xticks(unique_clusters)

    fig.suptitle(
        (
            f"Stage-1 Visualization: {csv_path.parent.name}"
            f" | min_nodes={args.min_nodes_to_show}"
            f" | silhouette={sil:.3f}"
            f" | DBI={dbi:.3f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[DONE] saved visualization: {out_path}")
    print(f"samples={len(cluster_ids)} clusters={len(unique_clusters)}")
    print(f"silhouette={sil:.6f} davies_bouldin={dbi:.6f}")


if __name__ == "__main__":
    main()
