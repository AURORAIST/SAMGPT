import argparse
import os
import random
import sys
import warnings
import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprompt import PrePrompt, pca_compression
from utils import process
from utils.dataset import load_dataset
from torch_geometric.loader import DataLoader


def torch_load_trusted(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_trusted(path: str, map_location=None):
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ego_nodes(adj_csr: sp.csr_matrix, node: int, hops: int) -> np.ndarray:
    frontier = {node}
    visited = {node}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            neighbors = adj_csr.indices[adj_csr.indptr[n] : adj_csr.indptr[n + 1]]
            next_frontier.update(neighbors.tolist())
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return np.array(sorted(visited), dtype=np.int64)


def compute_struct_signature(
    adj_csr: sp.csr_matrix, node_ids: np.ndarray, hops: int
) -> np.ndarray:
    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "networkx is required for structural signatures. Please install it."
        ) from exc

    signatures = []
    for n in node_ids:
        nodes = ego_nodes(adj_csr, int(n), hops)
        sub_adj = adj_csr[nodes][:, nodes]
        g = nx.from_scipy_sparse_array(sub_adj)
        if g.number_of_edges() > 0:
            g.remove_edges_from(nx.selfloop_edges(g))

        if g.number_of_nodes() == 0:
            signatures.append([0, 0, 0, 0, 0, 0])
            continue

        degrees = np.array([deg for _, deg in g.degree()], dtype=np.float32)
        avg_degree = float(degrees.mean()) if degrees.size else 0.0
        clustering = float(nx.average_clustering(g)) if g.number_of_edges() > 0 else 0.0
        density = float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0

        core_numbers = nx.core_number(g) if g.number_of_edges() > 0 else {n: 0 for n in g.nodes()}
        max_core = float(max(core_numbers.values())) if core_numbers else 0.0

        triangles = nx.triangles(g)
        avg_triangles = float(np.mean(list(triangles.values()))) if triangles else 0.0

        ego_size = float(g.number_of_nodes())
        signatures.append([
            avg_degree,
            clustering,
            density,
            max_core,
            avg_triangles,
            ego_size,
        ])

    return np.array(signatures, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a_n * b_n).sum(axis=-1)


def reduce_2d(x: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, init="random", random_state=None, n_jobs=1)
        return reducer.fit_transform(x)
    except Exception:
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=seed, init="pca")
        return reducer.fit_transform(x)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import pairwise_distances

    xy = np.concatenate([x, y], axis=0)
    dists = pairwise_distances(xy, xy, metric="euclidean")
    median = np.median(dists)
    sigma = median if median > 1e-9 else 1.0

    def k(a, b):
        d = pairwise_distances(a, b, metric="euclidean")
        return np.exp(-(d ** 2) / (2 * sigma ** 2))

    k_xx = k(x, x).mean()
    k_yy = k(y, y).mean()
    k_xy = k(x, y).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def train_linear_head(
    embeddings: torch.Tensor, labels: torch.Tensor, epochs: int = 200
) -> torch.nn.Module:
    in_dim = embeddings.size(1)
    num_classes = int(labels.max().item() + 1)
    head = torch.nn.Linear(in_dim, num_classes, bias=True).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(embeddings)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    return head


def grad_cosine(head: torch.nn.Module, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    head.zero_grad(set_to_none=True)
    logits = head(emb)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    grads = []
    for p in head.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
    return torch.cat(grads, dim=0)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("Visualize distribution contamination")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--pretrain_datasets",
        nargs="+",
        type=str,
        default=["Citeseer", "Pubmed", "Photo", "Computers", "FacebookPagePage", "LastFMAsia"],
    )
    parser.add_argument("--pretrain_method", type=str, default="GRAPHCL")
    parser.add_argument("--downstream_task", type=str, default="node")
    parser.add_argument("--combinetype", type=str, default="mul")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ablation_pre", type=str, default="all")
    parser.add_argument("--ablation_down", type=str, default="all")
    parser.add_argument("--unify_dim", type=int, default=50)
    parser.add_argument("--hid_units", type=int, default=256)
    parser.add_argument("--layers_num", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone", type=str, default="gcn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--graphId", nargs="+", type=int, default=[1])
    parser.add_argument("--support_split", type=int, default=0)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--ego_hops", type=int, default=1)
    parser.add_argument("--sample_per_domain", type=int, default=0)
    parser.add_argument("--topk_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--output_dir", type=str, default="../result/figs")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(args.gpu)

    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    current_dir = os.path.dirname(current_file_path)

    output_dir = os.path.abspath(os.path.join(current_dir, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "vis_debug.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Output dir: %s", output_dir)

    pretrain_dataset_str = "_" + "_".join(args.pretrain_datasets)
    set_name = (
        f"model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_"
        f"{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_"
        f"{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}"
    )
    save_dir = os.path.join(parent_directory, "checkpoints")
    save_name = os.path.join(save_dir, f"{set_name}.pkl")
    checkpoint_path = args.checkpoint or save_name
    if not os.path.exists(checkpoint_path):
        alt_dir = os.path.join(parent_directory, "checkpoint")
        alt_path = os.path.join(alt_dir, f"{set_name}.pkl")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
    logging.info("Checkpoint: %s", checkpoint_path)

    try:
        from torch_geometric.data.data import DataEdgeAttr

        torch.serialization.add_safe_globals([DataEdgeAttr])
    except Exception:
        logging.info("DataEdgeAttr not available for safe globals.")

    model = PrePrompt(
        args.unify_dim,
        args.hid_units,
        "prelu",
        len(args.pretrain_datasets) + len(args.graphId) - 1,
        args.layers_num,
        0.1,
        type_=args.combinetype,
        backbone=args.backbone,
        alpha=args.alpha,
        ablation=args.ablation_pre,
    ).cuda()
    model.load_state_dict(load_state_dict_trusted(checkpoint_path))
    model.eval()

    source_embeddings = []
    source_struct = []
    source_domain = []

    for dataset_name in args.pretrain_datasets:
        dataset = load_dataset(dataset_name)
        loader = DataLoader(dataset)
        for data in loader:
            features, adj = process.process_tu(data, data.x.shape[1])
            features = torch.FloatTensor(pca_compression(features, k=args.unify_dim)).cuda()
            adj_norm = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj_torch = process.sparse_mx_to_torch_sparse_tensor(adj_norm).cuda()
            with torch.no_grad():
                embeds, _ = model.embed(features, adj_torch, True, None, args.pretrain_method == "LP")
            embeds_np = embeds[0].detach().cpu().numpy()
            adj_csr = adj_norm.tocsr()

            node_ids = np.arange(adj_csr.shape[0])
            if args.sample_per_domain and len(node_ids) > args.sample_per_domain:
                node_ids = np.random.choice(node_ids, size=args.sample_per_domain, replace=False)
            struct = compute_struct_signature(adj_csr, node_ids, args.ego_hops)

            source_embeddings.append(embeds_np[node_ids])
            source_struct.append(struct)
            source_domain.extend([dataset_name] * len(node_ids))
            logging.info("Source domain %s: nodes=%d", dataset_name, len(node_ids))

    source_embeddings = np.concatenate(source_embeddings, axis=0)
    source_struct = np.concatenate(source_struct, axis=0)
    source_domain = np.array(source_domain)

    # Target dataset
    target_dataset = load_dataset(args.dataset)
    target_loader = DataLoader(target_dataset)
    for data in target_loader:
        t_features, t_adj = process.process_tu(data, data.x.shape[1])
        t_features = torch.FloatTensor(pca_compression(t_features, k=args.unify_dim)).cuda()
        t_adj_norm = process.normalize_adj(t_adj + sp.eye(t_adj.shape[0]))
        t_adj_torch = process.sparse_mx_to_torch_sparse_tensor(t_adj_norm).cuda()
        with torch.no_grad():
            t_embeds, _ = model.embed(t_features, t_adj_torch, True, None, args.pretrain_method == "LP")
        t_embeds_np = t_embeds[0].detach().cpu().numpy()
        t_adj_csr = t_adj_norm.tocsr()

        # Target support indices (few-shot)
        split_dir = os.path.join(
            parent_directory,
            "data",
            f"fewshot_{args.dataset.lower()}",
            f"{args.shot_num}-shot_{args.dataset.lower()}",
            str(args.support_split),
        )
        support_idx = torch.load(os.path.join(split_dir, "idx.pt")).cpu().numpy()
        support_labels = torch.load(os.path.join(split_dir, "labels.pt")).long().squeeze().cuda()

        t_support_embeds = t_embeds_np[support_idx]
        t_struct = compute_struct_signature(t_adj_csr, support_idx, args.ego_hops)

        # Full target (for some plots)
        target_node_ids = np.arange(t_adj_csr.shape[0])
        if args.sample_per_domain and len(target_node_ids) > args.sample_per_domain:
            target_node_ids = np.random.choice(
                target_node_ids, size=args.sample_per_domain, replace=False
            )
        t_struct_all = compute_struct_signature(t_adj_csr, target_node_ids, args.ego_hops)
        t_embeds_all = t_embeds_np[target_node_ids]

    # Normalize structural descriptors
    struct_all = np.concatenate([source_struct, t_struct_all], axis=0)
    struct_mean = struct_all.mean(axis=0, keepdims=True)
    struct_std = struct_all.std(axis=0, keepdims=True) + 1e-9
    source_struct_z = (source_struct - struct_mean) / struct_std
    t_struct_z = (t_struct_all - struct_mean) / struct_std
    t_support_struct_z = (t_struct - struct_mean) / struct_std

    # Selection score
    sem_centroid = t_support_embeds.mean(axis=0, keepdims=True)
    struct_centroid = t_support_struct_z.mean(axis=0, keepdims=True)
    sem_score = cosine_similarity(source_embeddings, sem_centroid)
    struct_score = cosine_similarity(source_struct_z, struct_centroid)
    score = sem_score + struct_score

    selected_idx_parts = []
    for d in args.pretrain_datasets:
        mask = source_domain == d
        domain_idx = np.where(mask)[0]
        if domain_idx.size == 0:
            continue
        k = max(1, int(domain_idx.size * args.topk_ratio))
        domain_scores = score[domain_idx]
        domain_selected = domain_idx[np.argsort(domain_scores)[-k:]]
        selected_idx_parts.append(domain_selected)
        logging.info("Topk per-domain %s: total=%d selected=%d", d, domain_idx.size, domain_selected.size)

    selected_idx = np.concatenate(selected_idx_parts) if selected_idx_parts else np.array([], dtype=np.int64)
    rejected_idx = np.setdiff1d(np.arange(len(score)), selected_idx)
    logging.info(
        "Topk ratio: %.3f | selected=%d | rejected=%d",
        args.topk_ratio,
        len(selected_idx),
        len(rejected_idx),
    )

    # UMAP 2x2
    import matplotlib.pyplot as plt
    import seaborn as sns

    sem_all = np.concatenate([source_embeddings, t_support_embeds], axis=0)
    sem_2d = reduce_2d(sem_all, args.seed)
    sem_source_2d = sem_2d[: len(source_embeddings)]
    sem_target_2d = sem_2d[len(source_embeddings) :]

    struct_all_2d = reduce_2d(np.concatenate([source_struct_z, t_support_struct_z], axis=0), args.seed)
    struct_source_2d = struct_all_2d[: len(source_struct_z)]
    struct_target_2d = struct_all_2d[len(source_struct_z) :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    domain_names = list(args.pretrain_datasets)
    palette = sns.color_palette("tab10", n_colors=len(domain_names))
    domain_color = {d: palette[i] for i, d in enumerate(domain_names)}

    # Semantic: source + target (color by domain)
    for d in domain_names:
        mask = source_domain == d
        axes[0, 0].scatter(
            sem_source_2d[mask, 0],
            sem_source_2d[mask, 1],
            s=6,
            c=[domain_color[d]],
            alpha=0.5,
            label=d,
        )
    axes[0, 0].scatter(
        sem_target_2d[:, 0], sem_target_2d[:, 1], s=20, c="#d27c2c", alpha=0.9, label="target"
    )
    axes[0, 0].set_title("Semantic: source + target")

    # Semantic: selected + target (selected colored by domain, rejected gray)
    axes[0, 1].scatter(
        sem_source_2d[rejected_idx, 0], sem_source_2d[rejected_idx, 1], s=4, c="#d1d5db", alpha=0.4
    )
    for d in domain_names:
        mask = (source_domain == d)
        mask_sel = mask.copy()
        mask_sel[rejected_idx] = False
        mask_sel = mask_sel & np.isin(np.arange(len(source_domain)), selected_idx)
        axes[0, 1].scatter(
            sem_source_2d[mask_sel, 0],
            sem_source_2d[mask_sel, 1],
            s=8,
            c=[domain_color[d]],
            alpha=0.8,
        )
    axes[0, 1].scatter(
        sem_target_2d[:, 0], sem_target_2d[:, 1], s=20, c="#d27c2c", alpha=0.9
    )
    axes[0, 1].set_title("Semantic: selected + target")

    # Structural: source + target (color by domain)
    for d in domain_names:
        mask = source_domain == d
        axes[1, 0].scatter(
            struct_source_2d[mask, 0],
            struct_source_2d[mask, 1],
            s=6,
            c=[domain_color[d]],
            alpha=0.5,
        )
    axes[1, 0].scatter(
        struct_target_2d[:, 0], struct_target_2d[:, 1], s=20, c="#d27c2c", alpha=0.9
    )
    axes[1, 0].set_title("Structural: source + target")

    # Structural: selected + target
    axes[1, 1].scatter(
        struct_source_2d[rejected_idx, 0], struct_source_2d[rejected_idx, 1], s=4, c="#d1d5db", alpha=0.4
    )
    for d in domain_names:
        mask = (source_domain == d)
        mask_sel = mask.copy()
        mask_sel[rejected_idx] = False
        mask_sel = mask_sel & np.isin(np.arange(len(source_domain)), selected_idx)
        axes[1, 1].scatter(
            struct_source_2d[mask_sel, 0],
            struct_source_2d[mask_sel, 1],
            s=8,
            c=[domain_color[d]],
            alpha=0.8,
        )
    axes[1, 1].scatter(
        struct_target_2d[:, 0], struct_target_2d[:, 1], s=20, c="#d27c2c", alpha=0.9
    )
    axes[1, 1].set_title("Structural: selected + target")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(loc="best", frameon=False, ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "umap_2x2.png"), dpi=300)
    plt.close(fig)

    # Structure signature heatmap
    domains = list(args.pretrain_datasets) + [args.dataset]
    struct_names = [
        "avg_degree",
        "clustering",
        "density",
        "k_core",
        "triangles",
        "ego_size",
    ]

    domain_struct = []
    offset = 0
    for dataset_name in args.pretrain_datasets:
        count = np.sum(source_domain == dataset_name)
        domain_struct.append(source_struct[offset : offset + count].mean(axis=0))
        offset += count
    domain_struct.append(t_struct_all.mean(axis=0))

    domain_struct = np.array(domain_struct)
    domain_struct_z = (domain_struct - domain_struct.mean(axis=0, keepdims=True)) / (
        domain_struct.std(axis=0, keepdims=True) + 1e-9
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(domain_struct_z, annot=False, cmap="viridis", ax=ax)
    ax.set_yticks(np.arange(len(domains)) + 0.5)
    ax.set_yticklabels(domains, rotation=0)
    ax.set_xticks(np.arange(len(struct_names)) + 0.5)
    ax.set_xticklabels(struct_names, rotation=30, ha="right")
    ax.set_title("Structure signature heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "structure_heatmap.png"), dpi=300)
    plt.close(fig)

    # Distance to target bar chart (semantic + structural)
    selected_sem = source_embeddings[selected_idx]
    selected_struct = source_struct_z[selected_idx]

    random_idx = np.random.choice(len(source_embeddings), size=len(selected_idx), replace=False)
    random_sem = source_embeddings[random_idx]
    random_struct = source_struct_z[random_idx]

    mmd_sem = [
        mmd_rbf(source_embeddings, t_support_embeds),
        mmd_rbf(random_sem, t_support_embeds),
        mmd_rbf(selected_sem, t_support_embeds),
    ]
    mmd_struct = [
        mmd_rbf(source_struct_z, t_support_struct_z),
        mmd_rbf(random_struct, t_support_struct_z),
        mmd_rbf(selected_struct, t_support_struct_z),
    ]
    logging.info("MMD semantic: %s", [f"{v:.4f}" for v in mmd_sem])
    logging.info("MMD structural: %s", [f"{v:.4f}" for v in mmd_struct])

    labels = ["Original", "Random", "Selected"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].bar(labels, mmd_sem, color=["#9aa0a6", "#9aa0a6", "#3b82f6"])
    axes[0].set_title("Semantic MMD")
    axes[1].bar(labels, mmd_struct, color=["#9aa0a6", "#9aa0a6", "#3b82f6"])
    axes[1].set_title("Structural MMD")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distance_mmd.png"), dpi=300)
    plt.close(fig)

    # Harmful ratio / gradient alignment
    t_support_embeds_t = torch.tensor(t_support_embeds, dtype=torch.float32).cuda()
    head = train_linear_head(t_support_embeds_t, support_labels)

    grad_target = grad_cosine(head, t_support_embeds_t, support_labels)
    grad_target = grad_target / (grad_target.norm() + 1e-9)

    def group_harmful_ratio(emb: np.ndarray, lbls: np.ndarray) -> float:
        emb_t = torch.tensor(emb, dtype=torch.float32).cuda()
        lbls_t = torch.tensor(lbls, dtype=torch.long).cuda()
        harmful = 0
        for i in range(emb_t.size(0)):
            g = grad_cosine(head, emb_t[i : i + 1], lbls_t[i : i + 1])
            g = g / (g.norm() + 1e-9)
            cos = torch.dot(g, grad_target).item()
            if cos < 0:
                harmful += 1
        return harmful / max(1, emb_t.size(0))

    # Map labels for source by nearest target support prototype
    support_proto = []
    for c in range(int(support_labels.max().item() + 1)):
        mask = support_labels == c
        support_proto.append(t_support_embeds_t[mask].mean(dim=0))
    support_proto = torch.stack(support_proto, dim=0)

    def pseudo_labels(emb: np.ndarray) -> np.ndarray:
        emb_t = torch.tensor(emb, dtype=torch.float32).cuda()
        sims = torch.cosine_similarity(emb_t[:, None, :], support_proto[None, :, :], dim=-1)
        return sims.argmax(dim=1).cpu().numpy()

    labels_source = pseudo_labels(source_embeddings)
    labels_random = labels_source[random_idx]
    labels_selected = labels_source[selected_idx]

    harmful_ratios = [
        group_harmful_ratio(source_embeddings, labels_source),
        group_harmful_ratio(random_sem, labels_random),
        group_harmful_ratio(selected_sem, labels_selected),
    ]
    logging.info("Harmful ratios: %s", [f"{v:.4f}" for v in harmful_ratios])

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.bar(labels, harmful_ratios, color=["#9aa0a6", "#9aa0a6", "#3b82f6"])
    ax.set_ylim(0, 1)
    ax.set_title("Harmful ratio (cos < 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "harmful_ratio.png"), dpi=300)
    plt.close(fig)

    logging.info("Saved figures to: %s", output_dir)


if __name__ == "__main__":
    main()
