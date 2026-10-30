#!/usr/bin/env python3
"""DGI-only structural-bucket negative analysis with k-hop similarity."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import structural_sampling  # noqa: E402


DEFAULT_DATASETS = [
    "Citeseer",
    "Computers",
    "FacebookPagePage",
    "LastFMAsia",
    "Photo",
    "Pubmed",
]


class KHopNeighborhood:
    def __init__(self, adj, hops):
        self.adj = adj.tocsr(copy=True)
        self.adj.data = np.ones_like(self.adj.data, dtype=np.int8)
        self.adj.setdiag(1)
        self.adj.eliminate_zeros()
        self.hops = hops
        self.cache = {}

    def get(self, node):
        node = int(node)
        if node in self.cache:
            return self.cache[node]
        seen = {node}
        frontier = {node}
        for _ in range(self.hops):
            nxt = set()
            for cur in frontier:
                start = self.adj.indptr[cur]
                end = self.adj.indptr[cur + 1]
                nxt.update(int(x) for x in self.adj.indices[start:end])
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
            if not frontier:
                break
        self.cache[node] = seen
        return seen


def load_adj(path):
    return torch.load(path, map_location="cpu", weights_only=False).tocsr()


def sample_nodes(n, max_nodes, rng):
    if max_nodes <= 0 or n <= max_nodes:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_nodes, replace=False))


def jaccard(neigh, a, b):
    na = neigh.get(a)
    nb = neigh.get(b)
    union = len(na | nb)
    return 1.0 if union == 0 else len(na & nb) / union


def perm_sims(neigh, nodes, perm):
    return np.fromiter((jaccard(neigh, int(i), int(perm[int(i)])) for i in nodes), dtype=np.float64)


def sampled_bucket_and_hard_sims(adj, neigh, nodes, bins, hops, min_sim, max_sim, candidates_per_node, rng):
    bucket_ids, feats = structural_sampling.structural_bucket_ids(adj, bins=bins, max_hops=hops)
    buckets = {}
    for idx, bucket_id in enumerate(bucket_ids):
        buckets.setdefault(int(bucket_id), []).append(idx)
    buckets = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    centers = structural_sampling.bucket_centers(bucket_ids, feats)

    bucket_vals = []
    hard_vals = []
    for i in nodes:
        i = int(i)
        ordered = sorted(
            buckets.keys(),
            key=lambda other: float(np.linalg.norm(centers[int(bucket_ids[i])] - centers[other])),
        )
        pool = []
        for bucket_id in ordered:
            candidates = buckets[bucket_id]
            candidates = candidates[candidates != i]
            if candidates.size == 0:
                continue
            candidates = candidates.copy()
            rng.shuffle(candidates)
            needed = candidates_per_node - len(pool)
            pool.extend(candidates[:needed].tolist())
            if len(pool) >= candidates_per_node:
                break

        if not pool:
            bucket_vals.append(0.0)
            hard_vals.append(0.0)
            continue

        bucket_j = int(pool[0])
        bucket_vals.append(jaccard(neigh, i, bucket_j))

        scored = [(j, jaccard(neigh, i, int(j))) for j in pool]
        valid = [(j, sim) for j, sim in scored if min_sim <= sim <= max_sim]
        if valid:
            valid.sort(key=lambda x: x[1], reverse=True)
            top = valid[: max(1, min(8, len(valid)))]
            hard_vals.append(float(top[int(rng.integers(0, len(top)))][1]))
            continue
        below = [(j, sim) for j, sim in scored if sim <= max_sim]
        if below:
            hard_vals.append(float(max(below, key=lambda x: x[1])[1]))
        else:
            hard_vals.append(float(min(scored, key=lambda x: x[1])[1]))

    return np.asarray(bucket_vals, dtype=np.float64), np.asarray(hard_vals, dtype=np.float64)


def summarize(dataset, metric, values):
    return {
        "dataset": dataset,
        "metric": metric,
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "p10": float(np.quantile(values, 0.10)) if values.size else np.nan,
        "p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="../cache")
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--min-sim", type=float, default=0.05)
    parser.add_argument("--max-sim", type=float, default=0.70)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    axes = axes.reshape(-1)
    hist_bins = np.linspace(0.0, 1.0, 51)

    random_means = []
    bucket_means = []
    hard_means = []
    random_zero = []
    bucket_zero = []
    hard_zero = []
    labels = []

    for ax, dataset in zip(axes, args.datasets):
        adj = load_adj(cache_dir / f"{dataset}_adj.pt")
        neigh = KHopNeighborhood(adj, hops=args.hops)
        nodes = sample_nodes(adj.shape[0], args.max_nodes, rng)
        random_perm = rng.permutation(adj.shape[0])
        bucket_vals, hard_vals = sampled_bucket_and_hard_sims(
            adj,
            neigh,
            nodes,
            args.bins,
            args.hops,
            args.min_sim,
            args.max_sim,
            args.candidates,
            rng,
        )

        random_vals = perm_sims(neigh, nodes, random_perm)

        rows.append(summarize(dataset, "DGI random shuffle", random_vals))
        rows.append(summarize(dataset, f"DGI structural bucket h{args.hops}", bucket_vals))
        rows.append(summarize(dataset, f"DGI structural hard h{args.hops}", hard_vals))
        random_means.append(float(np.mean(random_vals)))
        bucket_means.append(float(np.mean(bucket_vals)))
        hard_means.append(float(np.mean(hard_vals)))
        random_zero.append(float(np.mean(random_vals == 0)))
        bucket_zero.append(float(np.mean(bucket_vals == 0)))
        hard_zero.append(float(np.mean(hard_vals == 0)))
        labels.append(dataset)

        ax.hist(random_vals, bins=hist_bins, density=True, histtype="step", linewidth=1.8, color="#c53030", label="Random DGI shuffle")
        ax.hist(bucket_vals, bins=hist_bins, density=True, histtype="step", linewidth=1.8, color="#2b6cb0", label=f"Structural bucket DGI shuffle ({args.hops}-hop)")
        ax.hist(hard_vals, bins=hist_bins, density=True, histtype="step", linewidth=1.8, color="#2f855a", label=f"Structural hard DGI shuffle ({args.hops}-hop)")
        ax.set_title(dataset)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Structural similarity ({args.hops}-hop neighborhood Jaccard)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("DGI negative structural similarity: random vs structural samplers", y=0.955, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    density_path = out_dir / f"dgi_structural_bucket_density_h{args.hops}.png"
    fig.savefig(density_path, dpi=220)
    plt.close(fig)

    x = np.arange(len(labels))
    width = 0.36
    bar_fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    width = 0.25
    axes[0].bar(x - width, random_means, width, label="Random", color="#c53030")
    axes[0].bar(x, bucket_means, width, label="Structural bucket", color="#2b6cb0")
    axes[0].bar(x + width, hard_means, width, label="Structural hard", color="#2f855a")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel(f"Mean {args.hops}-hop structural similarity")
    axes[0].set_title("Mean similarity")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - width, random_zero, width, label="Random", color="#c53030")
    axes[1].bar(x, bucket_zero, width, label="Structural bucket", color="#2b6cb0")
    axes[1].bar(x + width, hard_zero, width, label="Structural hard", color="#2f855a")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Ratio of zero-similarity negatives")
    axes[1].set_title("How many negatives are still structure-unrelated")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    bar_fig.suptitle("DGI structural negative sampling v2", y=1.02)
    bar_fig.tight_layout()
    bar_path = out_dir / f"dgi_structural_bucket_bars_h{args.hops}.png"
    bar_fig.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close(bar_fig)

    csv_path = out_dir / f"dgi_structural_bucket_summary_h{args.hops}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "count", "mean", "median", "p10", "p90"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {density_path}")
    print(f"saved: {bar_path}")
    print(f"saved: {csv_path}")
    for rand, bucket, hard, z_rand, z_bucket, z_hard, dataset in zip(
        random_means,
        bucket_means,
        hard_means,
        random_zero,
        bucket_zero,
        hard_zero,
        labels,
    ):
        print(
            f"{dataset:16s} mean random={rand:.4f} bucket={bucket:.4f} hard={hard:.4f} "
            f"zero random={z_rand:.4f} bucket={z_bucket:.4f} hard={z_hard:.4f}"
        )


if __name__ == "__main__":
    main()
