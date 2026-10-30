from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train.stage_pipeline import Stage1Config, run_stage1  # type: ignore[reportMissingImports]

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


def resolve_feature_keys(csv_path: Path) -> list[str]:
    summary_path = csv_path.parent / "stage1_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            keys = payload.get("cluster_feature_names", None)
            if isinstance(keys, list) and len(keys) > 0:
                return [str(k) for k in keys]
        except Exception:
            pass
    return FEATURE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize stage1 structural encodings")
    parser.add_argument("--input-csv", type=str, default="", help="Path to structural_encodings.csv")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name for optional stage1 verification run")
    parser.add_argument("--data-root", type=str, default=str((PROJECT_ROOT / "data").resolve()), help="Dataset root")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str((PROJECT_ROOT / "result" / "stage_outputs" / "stage1").resolve()),
        help="Stage1 output root",
    )
    parser.add_argument("--verify-stage1", type=int, default=0, help="1: run stage1 first and use its csv")
    parser.add_argument("--max-subgraph-nodes", type=int, default=64)
    parser.add_argument("--subgraph-hop-n", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--encoding-mode", type=str, default="raw9", choices=["raw9", "decorrelated7"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove-selfloops", type=int, default=0)
    parser.add_argument("--use-log1p", type=int, default=1)
    parser.add_argument("--output-image", type=str, default="", help="Output image path")
    parser.add_argument("--min-nodes-to-show", type=int, default=1)
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data: dict[str, np.ndarray] = {}
    for key in arr.dtype.names:
        data[key] = np.asarray(arr[key])
    return data


def resolve_input_csv(args: argparse.Namespace) -> Path:
    if args.verify_stage1 == 1:
        if not args.dataset:
            raise ValueError("--dataset is required when --verify-stage1=1")
        cfg = Stage1Config(
            dataset=args.dataset,
            data_root=args.data_root,
            output_root=args.output_root,
            max_subgraph_nodes=args.max_subgraph_nodes,
            subgraph_hop_n=args.subgraph_hop_n,
            num_clusters=args.num_clusters,
            encoding_mode=args.encoding_mode,
            seed=args.seed,
            remove_selfloops=bool(args.remove_selfloops),
            use_log1p=bool(args.use_log1p),
        )
        run_stage1(cfg)
        return (Path(args.output_root).resolve() / args.dataset.lower() / "structural_encodings.csv").resolve()

    if not args.input_csv:
        raise ValueError("Provide --input-csv or set --verify-stage1=1 with --dataset")
    return Path(args.input_csv).resolve()


def main() -> None:
    args = parse_args()
    csv_path = resolve_input_csv(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.output_image).resolve() if args.output_image else csv_path.parent / "stage1_visualization.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_csv(csv_path)
    keys = [k for k in resolve_feature_keys(csv_path) if k in data]
    if len(keys) == 0:
        raise ValueError("No valid feature keys found for visualization")
    x = np.column_stack([data[k].astype(float) for k in keys])
    cluster_ids = data["cluster_id"].astype(int)
    num_nodes = data["num_nodes"].astype(float)

    mask = num_nodes >= float(args.min_nodes_to_show)
    x = x[mask]
    cluster_ids = cluster_ids[mask]

    if len(cluster_ids) == 0:
        raise ValueError("No samples left after filtering. Lower --min-nodes-to-show.")

    x_scaled = StandardScaler().fit_transform(x)
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
            f" | keys={','.join(keys)}"
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
