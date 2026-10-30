#!/usr/bin/env python3
"""Visualize structural-alignment bias in SAMGPT pretraining samples.

The script follows the official cache format:
  - *_adj.pt: raw scipy CSR adjacency used before normalization.
  - *_aug_adj.pt: stacked sparse torch tensor [orig, aug1, aug2] from build_aug().

It measures local structural similarity as Jaccard similarity between two
nodes' one-hop closed neighborhoods, or between the same node across two graph
views. The output density plot uses normalized histograms, so the y-axis is a
probability density.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch


DEFAULT_DATASETS = [
    "Citeseer",
    "Computers",
    "FacebookPagePage",
    "LastFMAsia",
    "Photo",
    "Pubmed",
]


def load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sparse_view_to_csr(stacked: torch.Tensor, view: int) -> sp.csr_matrix:
    """Convert one view from SAMGPT's stacked sparse COO tensor to CSR pattern."""
    coalesced = stacked.coalesce()
    idx = coalesced.indices().cpu().numpy()
    if idx.shape[0] == 3:
        keep = idx[0] == view
        rows = idx[1, keep]
        cols = idx[2, keep]
        shape = tuple(stacked.shape[1:])
    elif idx.shape[0] == 2:
        rows = idx[0]
        cols = idx[1]
        shape = tuple(stacked.shape)
    else:
        raise ValueError(f"Unexpected sparse tensor index shape: {idx.shape}")
    data = np.ones(rows.shape[0], dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=shape)


def closed_binary_csr(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    adj.data = np.ones_like(adj.data, dtype=np.int8)
    adj.setdiag(1)
    adj.eliminate_zeros()
    return adj


def sample_nodes(n: int, max_nodes: int, rng: np.random.Generator) -> np.ndarray:
    if max_nodes <= 0 or n <= max_nodes:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_nodes, replace=False))


def jaccard_rows(a: sp.csr_matrix, row_a: int, b: sp.csr_matrix, row_b: int) -> float:
    ai = a.indices[a.indptr[row_a] : a.indptr[row_a + 1]]
    bi = b.indices[b.indptr[row_b] : b.indptr[row_b + 1]]
    inter = np.intersect1d(ai, bi, assume_unique=True).size
    union = ai.size + bi.size - inter
    return 1.0 if union == 0 else inter / union


def view_similarity(orig: sp.csr_matrix, aug: sp.csr_matrix, nodes: np.ndarray) -> np.ndarray:
    return np.fromiter((jaccard_rows(orig, i, aug, i) for i in nodes), dtype=np.float64)


def shuffled_negative_similarity(
    neigh: sp.csr_matrix,
    nodes: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = neigh.shape[0]
    vals = []
    base = np.arange(n)
    for _ in range(repeats):
        perm = rng.permutation(base)
        vals.extend(jaccard_rows(neigh, i, neigh, int(perm[i])) for i in nodes)
    return np.asarray(vals, dtype=np.float64)


def lp_pair_similarity(
    raw_adj: sp.csr_matrix,
    neigh: sp.csr_matrix,
    nodes: np.ndarray,
    neg_per_node: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_adj.tocsr()
    n = raw.shape[0]
    all_nodes = np.arange(n)
    pos_vals = []
    neg_vals = []

    for i in nodes:
        nbrs = raw.indices[raw.indptr[i] : raw.indptr[i + 1]]
        nbrs = nbrs[nbrs != i]
        if nbrs.size:
            pos = int(rng.choice(nbrs))
        else:
            pos = int(i)
        pos_vals.append(jaccard_rows(neigh, int(i), neigh, pos))

        forbidden = np.zeros(n, dtype=bool)
        forbidden[nbrs] = True
        forbidden[int(i)] = True
        candidates = all_nodes[~forbidden]
        if candidates.size == 0:
            continue
        k = min(neg_per_node, candidates.size)
        negs = rng.choice(candidates, size=k, replace=False)
        neg_vals.extend(jaccard_rows(neigh, int(i), neigh, int(j)) for j in negs)

    return np.asarray(pos_vals, dtype=np.float64), np.asarray(neg_vals, dtype=np.float64)


def hard_nonedge_similarity(
    raw_adj: sp.csr_matrix,
    neigh: sp.csr_matrix,
    nodes: np.ndarray,
    candidates_per_node: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare random non-edge negatives with harder non-edges from sampled candidates."""
    raw = raw_adj.tocsr()
    n = raw.shape[0]
    all_nodes = np.arange(n)
    random_vals = []
    hard_vals = []

    for i in nodes:
        nbrs = raw.indices[raw.indptr[i] : raw.indptr[i + 1]]
        forbidden = np.zeros(n, dtype=bool)
        forbidden[nbrs] = True
        forbidden[int(i)] = True
        candidates = all_nodes[~forbidden]
        if candidates.size == 0:
            continue
        k = min(candidates_per_node, candidates.size)
        sampled = rng.choice(candidates, size=k, replace=False)
        sims = np.fromiter(
            (jaccard_rows(neigh, int(i), neigh, int(j)) for j in sampled),
            dtype=np.float64,
            count=k,
        )
        random_vals.append(float(sims[0]))
        hard_vals.append(float(np.max(sims)))

    return np.asarray(random_vals, dtype=np.float64), np.asarray(hard_vals, dtype=np.float64)


def add_density(ax, values: np.ndarray, label: str, color: str, bins: np.ndarray) -> None:
    if values.size == 0:
        return
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label=label,
        color=color,
    )


def summarize(dataset: str, metric: str, values: np.ndarray) -> dict[str, object]:
    return {
        "dataset": dataset,
        "metric": metric,
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "p10": float(np.quantile(values, 0.10)) if values.size else np.nan,
        "p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="../cache")
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--lp-neg-per-node", type=int, default=5)
    parser.add_argument("--hard-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    diagnostics = []

    ncols = 2
    nrows = int(np.ceil(len(args.datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    bins = np.linspace(0.0, 1.0, 51)
    colors = {
        "GraphCL edge aug": "#2b6cb0",
        "DGI shuffle negative": "#c53030",
        "LP positive edge": "#2f855a",
        "LP sampled non-edge": "#805ad5",
    }

    for ax, dataset in zip(axes, args.datasets):
        raw_adj_path = cache_dir / f"{dataset}_adj.pt"
        aug_adj_path = cache_dir / f"{dataset}_aug_adj.pt"
        if not raw_adj_path.exists() or not aug_adj_path.exists():
            ax.set_title(f"{dataset} (missing cache)")
            ax.axis("off")
            continue

        raw_adj = load_torch(raw_adj_path).tocsr()
        aug_stack = load_torch(aug_adj_path)
        orig_view = closed_binary_csr(sparse_view_to_csr(aug_stack, 0))
        aug1_view = closed_binary_csr(sparse_view_to_csr(aug_stack, 1))
        aug2_view = closed_binary_csr(sparse_view_to_csr(aug_stack, 2))
        nodes = sample_nodes(orig_view.shape[0], args.max_nodes, rng)

        aug_vals = np.concatenate(
            [view_similarity(orig_view, aug1_view, nodes), view_similarity(orig_view, aug2_view, nodes)]
        )
        dgi_vals = shuffled_negative_similarity(orig_view, nodes, args.shuffle_repeats, rng)
        lp_pos_vals, lp_neg_vals = lp_pair_similarity(
            raw_adj, orig_view, nodes, args.lp_neg_per_node, rng
        )
        random_nonedge_vals, hard_nonedge_vals = hard_nonedge_similarity(
            raw_adj, orig_view, nodes, args.hard_candidates, rng
        )

        metrics = [
            ("GraphCL edge aug", aug_vals),
            ("DGI shuffle negative", dgi_vals),
            ("LP positive edge", lp_pos_vals),
            ("LP sampled non-edge", lp_neg_vals),
        ]
        for label, vals in metrics:
            add_density(ax, vals, label, colors[label], bins)
            rows.append(summarize(dataset, label, vals))

        diagnostics.append(
            {
                "dataset": dataset,
                "GraphCL edge aug": aug_vals,
                "DGI shuffle negative": dgi_vals,
                "LP positive edge": lp_pos_vals,
                "LP sampled non-edge": lp_neg_vals,
                "Random sampled non-edge": random_nonedge_vals,
                "Hardest of sampled non-edges": hard_nonedge_vals,
            }
        )

        ax.set_title(f"{dataset} (sampled nodes={len(nodes)})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Structural similarity (1-hop neighborhood Jaccard)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)

    for ax in axes[len(args.datasets) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("SAMGPT pretraining structural similarity density", y=0.955, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    png_path = out_dir / "structural_similarity_density.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    csv_path = out_dir / "structural_similarity_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "count", "mean", "median", "p10", "p90"])
        writer.writeheader()
        writer.writerows(rows)

    diagnostic_rows = []
    for item in diagnostics:
        for metric, vals in item.items():
            if metric == "dataset":
                continue
            row = summarize(item["dataset"], metric, vals)
            diagnostic_rows.append(row)

    diag_csv_path = out_dir / "structural_bias_diagnostics.csv"
    with diag_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "count", "mean", "median", "p10", "p90"])
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    diag_fig, diag_axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metric_order = ["GraphCL edge aug", "DGI shuffle negative", "LP positive edge", "LP sampled non-edge"]
    box_data = [np.concatenate([item[m] for item in diagnostics]) for m in metric_order]
    diag_axes[0].boxplot(box_data, labels=["GraphCL\naug", "DGI\nshuffle", "LP\npositive", "LP\nnon-edge"], showfliers=False)
    diag_axes[0].set_ylabel("Structural similarity")
    diag_axes[0].set_title("Overall sample difficulty")
    diag_axes[0].grid(axis="y", alpha=0.2)

    datasets = [item["dataset"] for item in diagnostics]
    x = np.arange(len(datasets))
    width = 0.36
    graphcl_gap = [
        np.mean(item["GraphCL edge aug"]) - np.mean(item["DGI shuffle negative"]) for item in diagnostics
    ]
    lp_gap = [
        np.mean(item["LP positive edge"]) - np.mean(item["LP sampled non-edge"]) for item in diagnostics
    ]
    diag_axes[1].bar(x - width / 2, graphcl_gap, width, label="GraphCL aug - DGI shuffle", color="#2b6cb0")
    diag_axes[1].bar(x + width / 2, lp_gap, width, label="LP pos - LP non-edge", color="#2f855a")
    diag_axes[1].set_xticks(x, datasets, rotation=35, ha="right")
    diag_axes[1].set_ylabel("Mean similarity gap")
    diag_axes[1].set_title("Positive-negative structural gap")
    diag_axes[1].legend(frameon=False)
    diag_axes[1].grid(axis="y", alpha=0.2)

    random_means = [np.mean(item["Random sampled non-edge"]) for item in diagnostics]
    hard_means = [np.mean(item["Hardest of sampled non-edges"]) for item in diagnostics]
    diag_axes[2].bar(x - width / 2, random_means, width, label="Random non-edge", color="#c53030")
    diag_axes[2].bar(x + width / 2, hard_means, width, label=f"Hardest of {args.hard_candidates}", color="#805ad5")
    diag_axes[2].set_xticks(x, datasets, rotation=35, ha="right")
    diag_axes[2].set_ylabel("Mean structural similarity")
    diag_axes[2].set_title("Hard negatives exist but random sampling misses them")
    diag_axes[2].legend(frameon=False)
    diag_axes[2].grid(axis="y", alpha=0.2)

    diag_fig.suptitle("SAMGPT structural bias diagnostics", y=1.02, fontsize=14)
    diag_fig.tight_layout()
    diag_png_path = out_dir / "structural_bias_diagnostics.png"
    diag_fig.savefig(diag_png_path, dpi=220, bbox_inches="tight")
    plt.close(diag_fig)

    print(f"saved: {png_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {diag_png_path}")
    print(f"saved: {diag_csv_path}")
    for row in rows:
        print(
            f"{row['dataset']:16s} {row['metric']:22s} "
            f"mean={row['mean']:.4f} median={row['median']:.4f} "
            f"p10={row['p10']:.4f} p90={row['p90']:.4f} n={row['count']}"
        )


if __name__ == "__main__":
    main()
