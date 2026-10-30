from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize stage1 structural encodings")
    parser.add_argument(
        "--mode",
        type=str,
        default="stage1",
        choices=["stage1", "stage2-rank-coverage"],
        help="stage1: PCA scatter + cluster histogram; stage2-rank-coverage: rank-prefix structural coverage curves",
    )
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
    parser.add_argument("--stage2-pt", type=str, default="", help="Path to stage2 learnability_scores.pt")
    parser.add_argument("--graph-id", type=int, default=0, help="graph_id to plot rank-prefix curves for")
    parser.add_argument("--num-points", type=int, default=200, help="number of sampled prefix points on x-axis")
    parser.add_argument(
        "--stage2-top-ratio",
        type=float,
        default=-1.0,
        help="optional marker for top-ratio (e.g., 0.3). negative value means disabled",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output csv path for plotted curve values",
    )
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


def resolve_stage2_pt(args: argparse.Namespace, csv_path: Path) -> Path:
    if args.stage2_pt:
        return Path(args.stage2_pt).resolve()
    default_pt = csv_path.parent.parent.parent / "stage2" / csv_path.parent.name / "learnability_scores.pt"
    return default_pt.resolve()


def torch_load_trusted(path: Path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def build_prefix_points(n: int, num_points: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n <= num_points:
        return np.arange(1, n + 1, dtype=np.int64)
    pts = np.linspace(1, n, num=num_points, dtype=np.int64)
    pts = np.unique(pts)
    if pts[0] != 1:
        pts = np.insert(pts, 0, 1)
    if pts[-1] != n:
        pts = np.append(pts, n)
    return pts


def plot_stage2_rank_prefix_coverage(
    csv_path: Path,
    stage2_pt_path: Path,
    output_image: Path,
    graph_id: int,
    num_points: int,
    stage2_top_ratio: float,
    output_csv: Path | None,
) -> None:
    if not stage2_pt_path.exists():
        raise FileNotFoundError(f"stage2 pt not found: {stage2_pt_path}")

    data = load_csv(csv_path)
    csv_graph_ids = data["graph_id"].astype(np.int64)
    csv_subgraph_ids = data["subgraph_id"].astype(np.int64)
    csv_cluster_ids = data["cluster_id"].astype(np.int64)

    row_mask = csv_graph_ids == int(graph_id)
    if not np.any(row_mask):
        raise ValueError(f"No stage1 rows found for graph_id={graph_id} in {csv_path}")

    sg_to_cluster: dict[int, int] = {}
    for sg_id, c_id in zip(csv_subgraph_ids[row_mask], csv_cluster_ids[row_mask]):
        sg_to_cluster[int(sg_id)] = int(c_id)

    all_clusters = sorted(set(sg_to_cluster.values()))
    if len(all_clusters) == 0:
        raise ValueError("No clusters found for requested graph_id")

    cluster_to_pos = {cid: i for i, cid in enumerate(all_clusters)}
    full_counts = np.zeros(len(all_clusters), dtype=np.float64)
    for cid in sg_to_cluster.values():
        full_counts[cluster_to_pos[cid]] += 1.0
    full_prob = full_counts / max(1.0, np.sum(full_counts))

    score_pt = torch_load_trusted(stage2_pt_path)
    graph_ids = score_pt["graph_id"].cpu().numpy().astype(np.int64)
    subgraph_ids = score_pt["subgraph_id"].cpu().numpy().astype(np.int64)
    rank_desc = score_pt["rank_desc"].cpu().numpy().astype(np.int64)

    ranked_valid = []
    for idx in rank_desc.tolist():
        if graph_ids[idx] != int(graph_id):
            continue
        sg_id = int(subgraph_ids[idx])
        if sg_id in sg_to_cluster:
            ranked_valid.append(idx)

    n = len(ranked_valid)
    if n == 0:
        raise ValueError(f"No stage2 ranked samples for graph_id={graph_id}")

    prefix_points = build_prefix_points(n, max(2, int(num_points)))

    x_ratio = []
    cluster_coverage = []
    structure_mass_coverage = []
    structure_fit = []

    for k in prefix_points.tolist():
        chosen = ranked_valid[:k]
        prefix_cluster_counts = np.zeros(len(all_clusters), dtype=np.float64)
        for idx in chosen:
            cid = sg_to_cluster[int(subgraph_ids[idx])]
            prefix_cluster_counts[cluster_to_pos[cid]] += 1.0

        present = prefix_cluster_counts > 0
        cov = float(np.sum(present)) / float(len(all_clusters))
        mass_cov = float(np.sum(full_prob[present]))

        p_prefix = prefix_cluster_counts / max(1.0, np.sum(prefix_cluster_counts))
        tv = 0.5 * float(np.sum(np.abs(p_prefix - full_prob)))
        fit = 1.0 - tv

        x_ratio.append(float(k) / float(n))
        cluster_coverage.append(cov)
        structure_mass_coverage.append(mass_cov)
        structure_fit.append(fit)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_ratio, cluster_coverage, label="Cluster Coverage", linewidth=2.0)
    ax.plot(x_ratio, structure_mass_coverage, label="Structure Mass Coverage", linewidth=2.0)
    ax.plot(x_ratio, structure_fit, label="Structure Fit (1-TV)", linewidth=2.0)

    if 0.0 < stage2_top_ratio <= 1.0:
        ax.axvline(stage2_top_ratio, color="red", linestyle="--", linewidth=1.5, label=f"top_ratio={stage2_top_ratio:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Rank Prefix Ratio")
    ax.set_ylabel("Coverage / Fit")
    ax.set_title(f"Stage2 Rank-Prefix Structural Coverage | {csv_path.parent.name} | graph_id={graph_id}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_image, dpi=220)
    plt.close(fig)

    if output_csv is not None:
        lines = ["prefix_k,prefix_ratio,cluster_coverage,structure_mass_coverage,structure_fit"]
        for k, xr, cc, mc, sf in zip(prefix_points.tolist(), x_ratio, cluster_coverage, structure_mass_coverage, structure_fit):
            lines.append(f"{k},{xr:.8f},{cc:.8f},{mc:.8f},{sf:.8f}")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] saved stage2 rank-prefix curve: {output_image}")
    print(f"dataset={csv_path.parent.name} graph_id={graph_id} ranked_samples={n} clusters={len(all_clusters)}")
    if output_csv is not None:
        print(f"[DONE] saved curve csv: {output_csv}")


def main() -> None:
    args = parse_args()
    csv_path = resolve_input_csv(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    default_name = "stage1_visualization.png" if args.mode == "stage1" else "stage2_rank_prefix_coverage.png"
    out_path = Path(args.output_image).resolve() if args.output_image else csv_path.parent / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "stage1":
        data = load_csv(csv_path)
        x = np.column_stack([data[k].astype(float) for k in FEATURE_KEYS])
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
        return

    stage2_pt_path = resolve_stage2_pt(args, csv_path)
    curve_csv = Path(args.output_csv).resolve() if args.output_csv else out_path.with_suffix(".csv")
    plot_stage2_rank_prefix_coverage(
        csv_path=csv_path,
        stage2_pt_path=stage2_pt_path,
        output_image=out_path,
        graph_id=args.graph_id,
        num_points=args.num_points,
        stage2_top_ratio=args.stage2_top_ratio,
        output_csv=curve_csv,
    )


if __name__ == "__main__":
    main()
