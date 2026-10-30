from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from context_pipeline.stage1 import get_stage1_reference_graph, split_graph_to_subgraphs  # noqa: E402
from utils.dataset import load_dataset  # noqa: E402

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

LOG1P_FEATURE_IDX = [0, 1, 3, 4, 7, 8]
LOG1P_FEATURE_NAMES = [
    "num_nodes",
    "num_edges",
    "max_degree",
    "var_degree",
    "triangles",
    "diameter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate GMM clustering quality for stage1 structural encodings")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to structural_encodings.csv")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root for loading true labels")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name. Default: infer from stage1_summary.json")
    parser.add_argument("--num-components", type=int, default=8, help="Number of GMM components")
    parser.add_argument(
        "--min-components",
        type=int,
        default=0,
        help="Minimum components for sweep (>=2). If set with max-components, run sweep.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=0,
        help="Maximum components for sweep (>= min-components). If set with min-components, run sweep.",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "diag", "tied", "spherical"],
        help="GMM covariance type",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--use-log1p",
        type=int,
        default=-1,
        help="1 enable selected log1p, 0 disable, -1 auto from stage1_summary.json (default true)",
    )
    parser.add_argument(
        "--max-subgraph-nodes",
        type=int,
        default=64,
        help="Subgraph size used in stage1 (auto-read from summary when available)",
    )
    parser.add_argument(
        "--remove-selfloops",
        type=int,
        default=-1,
        help="1 remove self-loops, 0 keep, -1 auto-read from summary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: <csv_dir>/gmm_eval",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default="",
        help="Optional path to a Chinese font file (.ttf/.otf/.ttc)",
    )
    parser.add_argument(
        "--disable-umap",
        action="store_true",
        help="Disable UMAP plotting even if umap-learn is installed",
    )
    return parser.parse_args()


def configure_chinese_font(font_path: str) -> str:
    if font_path:
        p = Path(font_path).expanduser().resolve()
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            font_name = font_manager.FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return f"{font_name} ({p})"

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    chosen = [name for name in candidates if name in available]
    plt.rcParams["font.sans-serif"] = chosen if chosen else candidates
    plt.rcParams["axes.unicode_minus"] = False
    return chosen[0] if chosen else "<none-found>"


def load_csv_struct(csv_path: Path):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.size == 0:
        raise ValueError(f"Empty csv: {csv_path}")
    return np.atleast_1d(arr)


def _resolve_stage1_settings(csv_path: Path, args: argparse.Namespace) -> tuple[str, int, bool, bool]:
    dataset_name = args.dataset if args.dataset else csv_path.parent.name.capitalize()
    max_nodes = int(args.max_subgraph_nodes)
    remove_selfloops = bool(args.remove_selfloops) if args.remove_selfloops in (0, 1) else True
    use_log1p = bool(args.use_log1p) if args.use_log1p in (0, 1) else True

    summary_path = csv_path.parent / "stage1_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_name = str(summary.get("dataset", dataset_name))
            max_nodes = int(summary.get("max_subgraph_nodes", max_nodes))
            if args.remove_selfloops == -1:
                remove_selfloops = bool(summary.get("remove_selfloops", remove_selfloops))
            if args.use_log1p == -1:
                use_log1p = bool(summary.get("use_log1p", use_log1p))
        except Exception:
            pass

    if args.remove_selfloops in (0, 1):
        remove_selfloops = bool(args.remove_selfloops)
    if args.use_log1p in (0, 1):
        use_log1p = bool(args.use_log1p)

    return dataset_name, max_nodes, remove_selfloops, use_log1p


def _apply_log1p_selected(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for idx in LOG1P_FEATURE_IDX:
        out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
    return out


def _to_1d_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _majority_label(node_labels: np.ndarray) -> int:
    valid = node_labels[node_labels >= 0]
    if valid.size == 0:
        return -1
    counts = np.bincount(valid.astype(np.int64))
    return int(np.argmax(counts))


def build_subgraph_true_labels(
    csv_struct,
    dataset_name: str,
    data_root: str,
    max_subgraph_nodes: int,
    remove_selfloops: bool,
) -> np.ndarray:
    dataset = load_dataset(dataset_name, path=data_root)

    subgraph_label_map: dict[tuple[int, int], int] = {}
    for graph_id, data in enumerate(dataset):
        g = get_stage1_reference_graph(
            dataset_name=dataset_name,
            data_root=data_root,
            graph_id=int(graph_id),
            remove_selfloops=bool(remove_selfloops),
        )

        node_y = _to_1d_labels(data.y.cpu().numpy())
        subgraphs = split_graph_to_subgraphs(g, max_nodes=max_subgraph_nodes)
        for subgraph_id, sg in enumerate(subgraphs):
            nodes = np.array(list(sg.nodes()), dtype=np.int64)
            lbl = _majority_label(node_y[nodes]) if nodes.size > 0 else -1
            subgraph_label_map[(int(graph_id), int(subgraph_id))] = lbl

    y_true = np.full(len(csv_struct), -1, dtype=np.int64)
    for i, row in enumerate(csv_struct):
        key = (int(row["graph_id"]), int(row["subgraph_id"]))
        y_true[i] = int(subgraph_label_map.get(key, -1))
    return y_true


def clustering_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_vals, pred_idx = np.unique(y_pred, return_inverse=True)
    true_vals, true_idx = np.unique(y_true, return_inverse=True)
    mat = np.zeros((pred_vals.size, true_vals.size), dtype=np.int64)
    for p, t in zip(pred_idx, true_idx):
        mat[p, t] += 1

    cost = mat.max() - mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = mat[row_ind, col_ind].sum()
    return float(matched / y_true.size)


def save_assignments_csv(csv_struct, labels: np.ndarray, probs: np.ndarray, out_csv: Path) -> None:
    header = ["sample_id", "graph_id", "subgraph_id", "cluster_id", "cluster_prob"]
    lines = [",".join(header)]
    max_prob = probs.max(axis=1)
    for i, row in enumerate(csv_struct):
        lines.append(
            ",".join(
                [
                    str(int(row["sample_id"])),
                    str(int(row["graph_id"])),
                    str(int(row["subgraph_id"])),
                    str(int(labels[i])),
                    str(float(max_prob[i])),
                ]
            )
        )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scatter_by_labels(ax, emb2d: np.ndarray, labels: np.ndarray, title: str) -> None:
    valid_mask = labels >= 0
    if np.any(valid_mask):
        sc = ax.scatter(
            emb2d[valid_mask, 0],
            emb2d[valid_mask, 1],
            c=labels[valid_mask],
            cmap="tab10",
            s=18,
            alpha=0.85,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if np.any(~valid_mask):
        ax.scatter(
            emb2d[~valid_mask, 0],
            emb2d[~valid_mask, 1],
            c="#bbbbbb",
            s=14,
            alpha=0.6,
            edgecolors="none",
            label="未知标签",
        )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("维度1")
    ax.set_ylabel("维度2")


def save_embedding_comparison(
    embedding_name: str,
    emb2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray | None,
    out_png: Path,
    title_prefix: str,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _scatter_by_labels(ax1, emb2d, cluster_labels.astype(np.int64), f"{embedding_name} 按聚类标签着色")

    ax2 = fig.add_subplot(1, 2, 2)
    if true_labels is None:
        empty = np.full(cluster_labels.shape[0], -1, dtype=np.int64)
        _scatter_by_labels(ax2, emb2d, empty, f"{embedding_name} 按真实标签着色（不可用）")
    else:
        _scatter_by_labels(ax2, emb2d, true_labels.astype(np.int64), f"{embedding_name} 按真实标签着色")

    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_sweep_outputs(rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "num_components",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "aic",
        "bic",
        "nmi",
        "ari",
        "acc",
        "labeled_samples",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def plot_sweep_curves(rows: list[dict], out_png: Path) -> None:
    ks = np.array([int(r["num_components"]) for r in rows], dtype=np.int64)
    sil = np.array([float(r["silhouette"]) for r in rows], dtype=np.float64)
    dbi = np.array([float(r["davies_bouldin"]) for r in rows], dtype=np.float64)
    ch = np.array([float(r["calinski_harabasz"]) for r in rows], dtype=np.float64)
    aic = np.array([float(r["aic"]) for r in rows], dtype=np.float64)
    bic = np.array([float(r["bic"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(ks, sil, marker="o", label="Silhouette")
    ax.plot(ks, dbi, marker="s", label="DBI")
    ax.set_title("内部指标（越高越好/DBI越低越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(ks, ch, marker="o", color="#d62728", label="CH")
    ax.set_title("Calinski-Harabasz（越高越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(ks, aic, marker="o", color="#2ca02c", label="AIC")
    ax.plot(ks, bic, marker="s", color="#1f77b4", label="BIC")
    ax.set_title("信息准则（越低越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    has_supervised = any(np.isfinite(float(r.get("nmi", np.nan))) for r in rows)
    ax = axes[1, 1]
    if has_supervised:
        nmi = np.array([float(r.get("nmi", np.nan)) for r in rows], dtype=np.float64)
        ari = np.array([float(r.get("ari", np.nan)) for r in rows], dtype=np.float64)
        acc = np.array([float(r.get("acc", np.nan)) for r in rows], dtype=np.float64)
        ax.plot(ks, nmi, marker="o", label="NMI")
        ax.plot(ks, ari, marker="s", label="ARI")
        ax.plot(ks, acc, marker="^", label="ACC")
        ax.set_title("监督指标（越高越好）")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "无可用真实标签，跳过监督指标", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("监督指标")
    ax.set_xlabel("num_components")
    ax.grid(alpha=0.25)

    fig.suptitle("GMM 组件数扫描结果", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"input csv not found: {csv_path}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (csv_path.parent / "gmm_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    font_name = configure_chinese_font(args.font_path)
    csv_struct = load_csv_struct(csv_path)

    x_raw = np.column_stack([csv_struct[k].astype(float) for k in FEATURE_KEYS]).astype(np.float64)
    dataset_name, max_nodes, remove_selfloops, use_log1p = _resolve_stage1_settings(csv_path, args)

    x_trans = _apply_log1p_selected(x_raw) if use_log1p else x_raw.copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_trans)

    y_true = None
    try:
        y_true = build_subgraph_true_labels(
            csv_struct=csv_struct,
            dataset_name=dataset_name,
            data_root=args.data_root,
            max_subgraph_nodes=max_nodes,
            remove_selfloops=remove_selfloops,
        )
    except Exception as e:
        print(f"[WARN] failed to build true labels, skip NMI/ARI/ACC: {e}")

    if args.min_components > 0 or args.max_components > 0:
        if args.min_components < 2 or args.max_components < args.min_components:
            raise ValueError("Invalid sweep range. Expected: min-components>=2 and max-components>=min-components")
        k_values = list(range(int(args.min_components), int(args.max_components) + 1))
    else:
        if args.num_components < 2:
            raise ValueError("num-components must be >= 2")
        k_values = [int(args.num_components)]

    rows: list[dict] = []
    fit_cache: dict[int, tuple[np.ndarray, np.ndarray, GaussianMixture]] = {}
    for k in k_values:
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type=args.covariance_type,
            random_state=args.seed,
            reg_covar=1e-6,
        )
        labels = gmm.fit_predict(x_scaled)
        probs = gmm.predict_proba(x_scaled)
        fit_cache[int(k)] = (labels, probs, gmm)

        metrics = {
            "dataset": dataset_name,
            "num_samples": int(x_scaled.shape[0]),
            "num_components": int(k),
            "covariance_type": args.covariance_type,
            "seed": int(args.seed),
            "remove_selfloops": bool(remove_selfloops),
            "use_log1p": bool(use_log1p),
            "log1p_feature_names": LOG1P_FEATURE_NAMES,
            "silhouette": float(silhouette_score(x_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
            "aic": float(gmm.aic(x_scaled)),
            "bic": float(gmm.bic(x_scaled)),
        }

        if y_true is not None and np.any(y_true >= 0):
            mask = y_true >= 0
            yv = y_true[mask]
            pv = labels[mask]
            metrics["nmi"] = float(normalized_mutual_info_score(yv, pv))
            metrics["ari"] = float(adjusted_rand_score(yv, pv))
            metrics["acc"] = float(clustering_acc(yv, pv))
            metrics["labeled_samples"] = int(mask.sum())
        else:
            metrics["nmi"] = float("nan")
            metrics["ari"] = float("nan")
            metrics["acc"] = float("nan")
            metrics["labeled_samples"] = 0

        rows.append(metrics)

    # Pick the best K by BIC (lower is better), then export assignments/plots for that K.
    best_idx = int(np.argmin([float(r["bic"]) for r in rows]))
    best_metrics = rows[best_idx]
    best_k = int(best_metrics["num_components"])
    labels, probs, _best_gmm = fit_cache[best_k]

    out_json = out_dir / "gmm_metrics.json"
    out_assign = out_dir / "gmm_assignments.csv"
    out_diag = out_dir / "gmm_standardization_diagnostics.json"
    out_sweep_json = out_dir / "gmm_metrics_sweep.json"
    out_sweep_csv = out_dir / "gmm_metrics_sweep.csv"
    out_sweep_png = out_dir / "gmm_metrics_sweep.png"

    out_json.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    save_assignments_csv(csv_struct, labels, probs, out_assign)

    if len(rows) > 1:
        save_sweep_outputs(rows, out_sweep_json, out_sweep_csv)
        plot_sweep_curves(rows, out_sweep_png)

    diag = {
        "feature_names": FEATURE_KEYS,
        "raw_mean": x_raw.mean(axis=0).tolist(),
        "raw_std": x_raw.std(axis=0).tolist(),
        "scaled_mean": x_scaled.mean(axis=0).tolist(),
        "scaled_std": x_scaled.std(axis=0).tolist(),
        "use_log1p": bool(use_log1p),
        "log1p_feature_names": LOG1P_FEATURE_NAMES,
    }
    out_diag.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    pca = PCA(n_components=2, random_state=args.seed)
    pca2d = pca.fit_transform(x_scaled)
    out_pca = out_dir / "embedding_pca_cluster_vs_true.png"
    save_embedding_comparison(
        embedding_name="PCA",
        emb2d=pca2d,
        cluster_labels=labels,
        true_labels=y_true,
        out_png=out_pca,
        title_prefix=f"GMM 双着色对照 | {dataset_name} | K={best_k}",
    )

    out_umap = out_dir / "embedding_umap_cluster_vs_true.png"
    umap_ok = False
    if not args.disable_umap:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=args.seed)
            umap2d = reducer.fit_transform(x_scaled)
            save_embedding_comparison(
                embedding_name="UMAP",
                emb2d=umap2d,
                cluster_labels=labels,
                true_labels=y_true,
                out_png=out_umap,
                title_prefix=f"GMM 双着色对照 | {dataset_name} | K={best_k}",
            )
            umap_ok = True
        except Exception as e:
            print(f"[WARN] UMAP unavailable, skip UMAP plot: {e}")

    print("[DONE] GMM evaluation finished")
    print(f"input={csv_path}")
    print(f"output_metrics={out_json}")
    print(f"output_assignments={out_assign}")
    print(f"output_diag={out_diag}")
    print(f"output_pca={out_pca}")
    print(f"output_umap={out_umap if umap_ok else 'unavailable'}")
    if len(rows) > 1:
        print(f"output_sweep_json={out_sweep_json}")
        print(f"output_sweep_csv={out_sweep_csv}")
        print(f"output_sweep_png={out_sweep_png}")
    print(f"font={font_name}")
    print(
        f"best_k_by_bic={best_k}, "
        f"silhouette={best_metrics['silhouette']:.6f}, "
        f"dbi={best_metrics['davies_bouldin']:.6f}, "
        f"ch={best_metrics['calinski_harabasz']:.6f}, "
        f"nmi={best_metrics['nmi']}, ari={best_metrics['ari']}, acc={best_metrics['acc']}"
    )


if __name__ == "__main__":
    main()
