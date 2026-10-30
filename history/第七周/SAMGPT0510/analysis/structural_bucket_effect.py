#!/usr/bin/env python3
"""Compare random negatives with structural-bucket negatives."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import process, structural_sampling  # noqa: E402


DEFAULT_DATASETS = [
    "Citeseer",
    "Computers",
    "FacebookPagePage",
    "LastFMAsia",
    "Photo",
    "Pubmed",
]


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def closed_binary_csr(adj):
    csr = adj.tocsr(copy=True)
    csr.data = np.ones_like(csr.data, dtype=np.int8)
    csr.setdiag(1)
    csr.eliminate_zeros()
    return csr


class KHopNeighborhood:
    def __init__(self, adj, hops):
        self.adj = closed_binary_csr(adj)
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


def sample_nodes(n, max_nodes, rng):
    if max_nodes <= 0 or n <= max_nodes:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_nodes, replace=False))


def jaccard_rows(neigh, row_a, row_b):
    ai = neigh.get(row_a)
    bi = neigh.get(row_b)
    inter = len(ai & bi)
    union = len(ai | bi)
    return 1.0 if union == 0 else inter / union


def perm_sims(neigh, nodes, perm):
    return np.fromiter((jaccard_rows(neigh, int(i), int(perm[i])) for i in nodes), dtype=np.float64)


def tuple_negative_sims(neigh, nodes, tuples):
    vals = []
    tuples = tuples.numpy() if torch.is_tensor(tuples) else tuples
    for i in nodes:
        vals.extend(jaccard_rows(neigh, int(i), int(j)) for j in tuples[int(i), 1:])
    return np.asarray(vals, dtype=np.float64)


def random_lp_tuples(adj, n):
    import preprompt

    return preprompt.prompt_pretrain_sample(adj, n)


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


def plot_density(ax, values, label, color, bins):
    ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8, label=label, color=color)


def combined_domain_ids(adjs):
    ids = []
    for domain, adj in enumerate(adjs):
        ids.extend([domain] * adj.shape[0])
    return np.asarray(ids, dtype=np.int64)


def sampled_random_nonedge_stats(neigh, adj, nodes, negatives, domain_ids, rng):
    csr = adj.tocsr()
    n = csr.shape[0]
    sims = []
    cross = []
    for i in nodes:
        nbrs = set(csr.indices[csr.indptr[int(i)] : csr.indptr[int(i) + 1]].tolist())
        nbrs.add(int(i))
        picked = []
        while len(picked) < negatives:
            j = int(rng.integers(0, n))
            if j in nbrs:
                continue
            picked.append(j)
        sims.extend(jaccard_rows(neigh, int(i), j) for j in picked)
        cross.extend(domain_ids[int(i)] != domain_ids[picked])
    return np.asarray(sims, dtype=np.float64), float(np.mean(cross))


def sampled_bucket_nonedge_stats(neigh, adj, nodes, negatives, domain_ids, bins, rng, hops):
    csr = adj.tocsr()
    bucket_ids, feats = structural_sampling.structural_bucket_ids(csr, bins=bins, max_hops=hops)
    buckets = {}
    for idx, bucket_id in enumerate(bucket_ids):
        buckets.setdefault(int(bucket_id), []).append(idx)
    buckets = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    centers = structural_sampling.bucket_centers(bucket_ids, feats)
    n = csr.shape[0]
    whole = np.arange(n)
    sims = []
    cross = []

    for i in nodes:
        nbrs = set(csr.indices[csr.indptr[int(i)] : csr.indptr[int(i) + 1]].tolist())
        nbrs.add(int(i))
        picked = []
        ordered = sorted(
            buckets.keys(),
            key=lambda other: float(np.linalg.norm(centers[int(bucket_ids[int(i)])] - centers[other])),
        )
        for bucket_id in ordered:
            candidates = [int(c) for c in buckets[bucket_id] if int(c) not in nbrs]
            if not candidates:
                continue
            rng.shuffle(candidates)
            picked.extend(candidates[: negatives - len(picked)])
            if len(picked) >= negatives:
                break
        while len(picked) < negatives:
            j = int(rng.choice(whole))
            if j not in nbrs:
                picked.append(j)
        sims.extend(jaccard_rows(neigh, int(i), j) for j in picked)
        cross.extend(domain_ids[int(i)] != domain_ids[picked])
    return np.asarray(sims, dtype=np.float64), float(np.mean(cross))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="../cache")
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    axes = axes.reshape(-1)
    bins = np.linspace(0.0, 1.0, 51)
    colors = {
        "Random DGI shuffle": "#c53030",
        "Structural-bucket DGI shuffle": "#2b6cb0",
        "Random LP non-edge": "#805ad5",
        "Structural-bucket LP non-edge": "#2f855a",
    }

    raw_adjs = []
    for ax, dataset in zip(axes, args.datasets):
        adj = load_torch(cache_dir / f"{dataset}_adj.pt").tocsr()
        raw_adjs.append(adj)
        neigh = KHopNeighborhood(adj, hops=args.hops)
        nodes = sample_nodes(adj.shape[0], args.max_nodes, rng)

        random_perm = rng.permutation(adj.shape[0])
        bucket_perm = structural_sampling.structure_aware_permutation(
            adj,
            bins=args.bins,
            max_hops=args.hops,
        ).numpy()
        local_domain_ids = np.zeros(adj.shape[0], dtype=np.int64)
        random_lp_sims, _ = sampled_random_nonedge_stats(
            neigh,
            adj,
            nodes,
            args.negatives,
            local_domain_ids,
            rng,
        )
        bucket_lp_sims, _ = sampled_bucket_nonedge_stats(
            neigh,
            adj,
            nodes,
            args.negatives,
            local_domain_ids,
            args.bins,
            rng,
            args.hops,
        )

        metrics = {
            "Random DGI shuffle": perm_sims(neigh, nodes, random_perm),
            "Structural-bucket DGI shuffle": perm_sims(neigh, nodes, bucket_perm),
            "Random LP non-edge": random_lp_sims,
            "Structural-bucket LP non-edge": bucket_lp_sims,
        }

        for label, vals in metrics.items():
            plot_density(ax, vals, label, colors[label], bins)
            rows.append(summarize(dataset, label, vals))

        ax.set_title(dataset)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Structural similarity ({args.hops}-hop neighborhood Jaccard)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Effect of structural-bucket negative sampling", y=0.955, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    png_path = out_dir / f"structural_bucket_effect_density_h{args.hops}.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    combined_adj = process.combine_dataset_list_sp(raw_adjs)
    combined_neigh = KHopNeighborhood(combined_adj, hops=args.hops)
    domain_ids = combined_domain_ids(raw_adjs)
    combined_nodes = sample_nodes(combined_adj.shape[0], args.max_nodes, rng)

    def combined_stats(name, sims, cross_ratio):
        row = summarize("combined_multidomain", name, sims)
        row["cross_domain_ratio"] = cross_ratio
        return row

    random_combined_sims, random_cross_ratio = sampled_random_nonedge_stats(
        combined_neigh,
        combined_adj,
        combined_nodes,
        args.negatives,
        domain_ids,
        rng,
    )
    bucket_combined_sims, bucket_cross_ratio = sampled_bucket_nonedge_stats(
        combined_neigh,
        combined_adj,
        combined_nodes,
        args.negatives,
        domain_ids,
        args.bins,
        rng,
        args.hops,
    )

    combined_rows = [
        combined_stats("Random LP non-edge", random_combined_sims, random_cross_ratio),
        combined_stats("Structural-bucket LP non-edge", bucket_combined_sims, bucket_cross_ratio),
    ]

    csv_path = out_dir / f"structural_bucket_effect_summary_h{args.hops}.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["dataset", "metric", "count", "mean", "median", "p10", "p90", "cross_domain_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["cross_domain_ratio"] = ""
            writer.writerow(row)
        writer.writerows(combined_rows)

    by_metric = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row["mean"])
    dataset_labels = args.datasets
    x = np.arange(len(dataset_labels))
    width = 0.35
    bar_fig, bar_axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bar_axes[0].bar(x - width / 2, by_metric["Random DGI shuffle"], width, label="Random", color="#c53030")
    bar_axes[0].bar(
        x + width / 2,
        by_metric["Structural-bucket DGI shuffle"],
        width,
        label="Structural bucket",
        color="#2b6cb0",
    )
    bar_axes[0].set_title("DGI shuffle negatives")
    bar_axes[0].set_ylabel("Mean structural similarity")
    bar_axes[0].set_xticks(x, dataset_labels, rotation=35, ha="right")
    bar_axes[0].legend(frameon=False)
    bar_axes[0].grid(axis="y", alpha=0.2)

    bar_axes[1].bar(x - width / 2, by_metric["Random LP non-edge"], width, label="Random", color="#805ad5")
    bar_axes[1].bar(
        x + width / 2,
        by_metric["Structural-bucket LP non-edge"],
        width,
        label="Structural bucket",
        color="#2f855a",
    )
    bar_axes[1].set_title("LP non-edge negatives")
    bar_axes[1].set_ylabel("Mean structural similarity")
    bar_axes[1].set_xticks(x, dataset_labels, rotation=35, ha="right")
    bar_axes[1].legend(frameon=False)
    bar_axes[1].grid(axis="y", alpha=0.2)

    bar_axes[2].bar(
        ["Random\nLP", "Structural\nbucket LP"],
        [combined_rows[0]["cross_domain_ratio"], combined_rows[1]["cross_domain_ratio"]],
        color=["#805ad5", "#2f855a"],
    )
    bar_axes[2].set_ylim(0, 1)
    bar_axes[2].set_title("Combined multi-domain negatives")
    bar_axes[2].set_ylabel("Cross-domain ratio")
    bar_axes[2].grid(axis="y", alpha=0.2)
    bar_fig.suptitle("Structural-bucket sampler raises structural relevance while preserving multi-domain negatives", y=1.02)
    bar_fig.tight_layout()
    bar_png_path = out_dir / f"structural_bucket_effect_bars_h{args.hops}.png"
    bar_fig.savefig(bar_png_path, dpi=220, bbox_inches="tight")
    plt.close(bar_fig)

    print(f"saved: {png_path}")
    print(f"saved: {bar_png_path}")
    print(f"saved: {csv_path}")
    for row in rows + combined_rows:
        suffix = ""
        if "cross_domain_ratio" in row:
            suffix = f" cross_domain_ratio={row['cross_domain_ratio']:.4f}"
        print(
            f"{row['dataset']:20s} {row['metric']:34s} "
            f"mean={row['mean']:.4f} median={row['median']:.4f} "
            f"p10={row['p10']:.4f} p90={row['p90']:.4f}{suffix}"
        )


if __name__ == "__main__":
    main()
