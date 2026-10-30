"""Structure-aware negative sampling utilities.

The samplers here keep the original SAMGPT behavior opt-in. They group nodes by
simple structural-role signatures, then sample negatives from the same or nearby
role buckets instead of from the whole graph uniformly at random.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch


def _binary_csr(adj):
    csr = adj.tocsr(copy=True)
    csr.data = np.ones_like(csr.data, dtype=np.float64)
    csr.eliminate_zeros()
    return csr


def _reachability_counts(csr, max_hops):
    base = _binary_csr(csr)
    base.setdiag(0)
    base.eliminate_zeros()
    reach = base.copy()
    frontier = base.copy()
    counts = [np.asarray(reach.sum(axis=1)).reshape(-1).astype(np.float64)]

    for _ in range(2, max_hops + 1):
        frontier = frontier @ base
        frontier.data = np.ones_like(frontier.data, dtype=np.float64)
        frontier.setdiag(0)
        frontier.eliminate_zeros()
        reach = reach + frontier
        reach.data = np.ones_like(reach.data, dtype=np.float64)
        reach.eliminate_zeros()
        counts.append(np.asarray(reach.sum(axis=1)).reshape(-1).astype(np.float64))
    return counts


def structural_role_features(adj, max_hops=3):
    """Return lightweight structural signatures for each node.

    Features are intentionally local and cheap:
    - degree percentile
    - log degree
    - log mean neighbor degree
    - log std of neighbor degree
    - log cumulative 1/2/3-hop reachable node counts

    Percentile/log scaling makes buckets more comparable across domains whose
    raw density and node count differ.
    """
    csr = _binary_csr(adj)
    n = csr.shape[0]
    deg = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float64)
    safe_deg = np.maximum(deg, 1.0)

    neigh_deg_mean = csr.dot(deg) / safe_deg
    neigh_deg_sq_mean = csr.dot(deg * deg) / safe_deg
    neigh_deg_std = np.sqrt(np.maximum(neigh_deg_sq_mean - neigh_deg_mean * neigh_deg_mean, 0.0))

    order = np.argsort(deg, kind="mergesort")
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.linspace(0.0, 1.0, n, endpoint=True) if n > 1 else 0.0

    hop_counts = _reachability_counts(csr, max_hops=max_hops)

    feats = np.column_stack(
        [
            rank,
            np.log1p(deg),
            np.log1p(neigh_deg_mean),
            np.log1p(neigh_deg_std),
            *[np.log1p(counts) for counts in hop_counts],
        ]
    )
    return feats


def structural_bucket_ids(adj, bins=5, max_hops=3):
    """Quantize structural signatures into bucket ids."""
    feats = structural_role_features(adj, max_hops=max_hops)
    n, dim = feats.shape
    bucket_parts = []

    for col in range(dim):
        values = feats[:, col]
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        quantiles = np.unique(quantiles)
        bucket_parts.append(np.digitize(values, quantiles, right=False))

    parts = np.vstack(bucket_parts).T.astype(np.int64)
    ids = np.zeros(n, dtype=np.int64)
    for col in range(parts.shape[1]):
        ids = ids * bins + parts[:, col]
    return ids, feats


def _bucket_members(bucket_ids):
    buckets = defaultdict(list)
    for idx, bucket_id in enumerate(bucket_ids):
        buckets[int(bucket_id)].append(idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}


def _nearest_bucket_order(bucket_id, buckets, bucket_centers):
    center = bucket_centers[bucket_id]
    ordered = sorted(
        buckets.keys(),
        key=lambda other: float(np.linalg.norm(center - bucket_centers[other])),
    )
    return ordered


def bucket_centers(bucket_ids, feats):
    centers = {}
    for bucket_id, members in _bucket_members(bucket_ids).items():
        centers[bucket_id] = feats[members].mean(axis=0)
    return centers


class KHopNeighborhood:
    """Lazy k-hop closed-neighborhood cache for candidate scoring."""

    def __init__(self, adj, hops):
        self.adj = _binary_csr(adj)
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


def khop_jaccard(neighborhood, node_a, node_b):
    a = neighborhood.get(node_a)
    b = neighborhood.get(node_b)
    union = len(a | b)
    return 1.0 if union == 0 else len(a & b) / union


def permutation_diagnostics(adj, perm, max_hops=3, max_nodes=3000, rng=None):
    """Summarize k-hop structural similarity induced by a DGI shuffle permutation."""
    if rng is None:
        rng = np.random
    perm = perm.detach().cpu().numpy() if torch.is_tensor(perm) else np.asarray(perm)
    n = adj.shape[0]
    if max_nodes > 0 and n > max_nodes:
        nodes = np.sort(rng.choice(np.arange(n), size=max_nodes, replace=False))
    else:
        nodes = np.arange(n)

    neighborhood = KHopNeighborhood(adj, hops=max_hops)
    values = np.asarray([khop_jaccard(neighborhood, int(i), int(perm[int(i)])) for i in nodes], dtype=np.float64)
    return {
        "sampled_nodes": int(nodes.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "p10": float(np.quantile(values, 0.10)) if values.size else np.nan,
        "p90": float(np.quantile(values, 0.90)) if values.size else np.nan,
        "zero_ratio": float(np.mean(values == 0.0)) if values.size else np.nan,
    }


def structure_aware_permutation(adj, bins=5, rng=None, max_hops=3):
    """Build a feature-shuffle permutation that preserves structural buckets."""
    if rng is None:
        rng = np.random
    bucket_ids, feats = structural_bucket_ids(adj, bins=bins, max_hops=max_hops)
    buckets = _bucket_members(bucket_ids)
    centers = bucket_centers(bucket_ids, feats)
    n = adj.shape[0]
    perm = np.arange(n, dtype=np.int64)

    for bucket_id, members in buckets.items():
        if members.size > 1:
            shuffled = members.copy()
            rng.shuffle(shuffled)
            if np.any(shuffled == members):
                shuffled = np.roll(shuffled, 1)
            perm[members] = shuffled
            continue

        i = int(members[0])
        for other_bucket in _nearest_bucket_order(bucket_id, buckets, centers):
            candidates = buckets[other_bucket]
            candidates = candidates[candidates != i]
            if candidates.size:
                perm[i] = int(rng.choice(candidates))
                break

    return torch.tensor(perm, dtype=torch.long)


def structure_hard_permutation(
    adj,
    bins=5,
    rng=None,
    max_hops=3,
    min_sim=0.05,
    max_sim=0.70,
    candidates_per_node=128,
    fallback_random_tries=512,
):
    """Build a DGI shuffle permutation with bucketed semi-hard negatives.

    Candidates are drawn from the anchor's structural bucket and nearby buckets,
    then scored by k-hop Jaccard. The sampler prefers candidates inside
    [min_sim, max_sim]. max_sim is a hard upper bound: if no candidate satisfies
    it, the sampler falls back to a low-similarity candidate rather than a
    possible false negative above max_sim.
    """
    if rng is None:
        rng = np.random

    csr = _binary_csr(adj)
    n = csr.shape[0]
    bucket_ids, feats = structural_bucket_ids(csr, bins=bins, max_hops=max_hops)
    buckets = _bucket_members(bucket_ids)
    centers = bucket_centers(bucket_ids, feats)
    neighborhood = KHopNeighborhood(csr, hops=max_hops)
    perm = np.arange(n, dtype=np.int64)

    for i in range(n):
        candidate_pool = []
        for bucket_id in _nearest_bucket_order(int(bucket_ids[i]), buckets, centers):
            candidates = buckets[bucket_id]
            candidates = candidates[candidates != i]
            if candidates.size == 0:
                continue
            candidates = candidates.copy()
            rng.shuffle(candidates)
            needed = candidates_per_node - len(candidate_pool)
            candidate_pool.extend(candidates[:needed].tolist())
            if len(candidate_pool) >= candidates_per_node:
                break

        if not candidate_pool:
            continue

        scored = [(j, khop_jaccard(neighborhood, i, j)) for j in candidate_pool]
        valid = [(j, sim) for j, sim in scored if min_sim <= sim <= max_sim]
        if valid:
            # Prefer the upper half of the allowed interval without forcing
            # near-duplicates all the time.
            valid.sort(key=lambda x: x[1], reverse=True)
            top = valid[: max(1, min(8, len(valid)))]
            perm[i] = int(top[int(rng.randint(len(top)))][0])
            continue

        low = [(j, sim) for j, sim in scored if sim < min_sim]
        if low:
            perm[i] = int(max(low, key=lambda x: x[1])[0])
            continue

        fallback = None
        fallback_sim = -1.0
        tried = set(candidate_pool)
        for _ in range(fallback_random_tries):
            j = int(rng.randint(n))
            if j == i or j in tried:
                continue
            tried.add(j)
            sim = khop_jaccard(neighborhood, i, j)
            if sim <= max_sim and sim > fallback_sim:
                fallback = j
                fallback_sim = sim
                if sim >= min_sim:
                    break

        if fallback is not None:
            perm[i] = int(fallback)
            continue

        # Extremely dense/local graphs may have no sampled candidate below the
        # upper bound. Pick the least similar observed candidate as the safest
        # available fallback and let diagnostics expose the residual risk.
        perm[i] = int(min(scored, key=lambda x: x[1])[0])

    return torch.tensor(perm, dtype=torch.long)


def prompt_pretrain_sample_structural_bucket(adj, n, bins=5, max_hops=3):
    """LP/splitLP tuples with structure-bucket negatives.

    Positives follow SAMGPT's original rule: one sampled nonzero neighbor, or the
    anchor itself for isolated nodes. Negatives are non-neighbors selected from
    the anchor's structural bucket first, then nearby structural buckets.
    """
    csr = _binary_csr(adj)
    nodenum = csr.shape[0]
    indices = csr.indices
    indptr = csr.indptr
    whole = np.arange(nodenum)
    res = np.zeros((nodenum, 1 + n), dtype=np.int64)

    bucket_ids, feats = structural_bucket_ids(csr, bins=bins, max_hops=max_hops)
    buckets = _bucket_members(bucket_ids)
    centers = bucket_centers(bucket_ids, feats)

    for i in range(nodenum):
        nonzero = indices[indptr[i] : indptr[i + 1]]
        nonzero = nonzero[nonzero != i]
        nonzero_shuf = nonzero.copy()
        np.random.shuffle(nonzero_shuf)
        res[i, 0] = int(nonzero_shuf[0]) if nonzero_shuf.size else i

        forbidden = set(nonzero.tolist())
        forbidden.add(i)

        negs = []
        for bucket_id in _nearest_bucket_order(int(bucket_ids[i]), buckets, centers):
            candidates = buckets[bucket_id]
            candidates = np.asarray([c for c in candidates if int(c) not in forbidden], dtype=np.int64)
            if candidates.size == 0:
                continue
            candidates = candidates.copy()
            np.random.shuffle(candidates)
            needed = n - len(negs)
            negs.extend(candidates[:needed].tolist())
            if len(negs) >= n:
                break

        if len(negs) < n:
            fallback = np.asarray([c for c in whole if int(c) not in forbidden], dtype=np.int64)
            if fallback.size == 0:
                fallback = whole[whole != i]
            fallback = fallback.copy()
            np.random.shuffle(fallback)
            needed = n - len(negs)
            if fallback.size:
                extra = np.resize(fallback, needed)
                negs.extend(extra.tolist())

        res[i, 1 : 1 + n] = np.asarray(negs[:n], dtype=np.int64)

    return torch.tensor(res.astype(int))
