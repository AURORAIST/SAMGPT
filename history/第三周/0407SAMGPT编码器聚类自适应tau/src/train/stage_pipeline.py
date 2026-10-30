from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from layers.stage1_output_writer import write_stage1_outputs  # type: ignore[reportMissingImports]
from layers.stage1_subgraph_builder import (
    get_stage1_reference_graph,
    iter_dataset_graphs,
    split_graph_to_subgraphs,
)
from layers.stage2_output_writer import write_stage2_outputs  # type: ignore[reportMissingImports]
from layers.stage2_pool_builder import build_representative_pool, load_table  # type: ignore[reportMissingImports]
from models.stage1_cluster import fit_clusters
from models.stage2_learnability_model import FEATURE_KEYS, compute_learnability  # type: ignore[reportMissingImports]
from utils import process  # type: ignore[reportMissingImports]
from utils.dataset import load_dataset  # type: ignore[reportMissingImports]


@dataclass
class Stage1Config:
    dataset: str
    data_root: str
    output_root: str
    max_subgraph_nodes: int = 64
    num_clusters: int = 8
    seed: int = 0
    remove_selfloops: bool = True
    use_log1p: bool = True
    subgraph_hop_n: int = 2
    encoding_mode: str = "raw9"
    cluster_method: str = "gmm"


@dataclass
class Stage2Config:
    input_csv: str
    output_dir: str
    top_m_per_cluster: int = 5
    top_p_percent: float = 0.3
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    center_path: str = ""
    dataset_name: str = ""
    data_root: str = ""
    max_subgraph_nodes: int = 64
    subgraph_hop_n: int = 2
    remove_selfloops: bool = True
    pca_dim: int = 50
    write_decoded_subgraphs: bool = True


def _safe_triangle_count(graph: nx.Graph) -> float:
    triangle_map = nx.triangles(graph)
    return float(sum(triangle_map.values()) / 3.0)


def _safe_diameter(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0

    if nx.is_connected(graph):
        return float(nx.diameter(graph))

    diameters = []
    for nodes in nx.connected_components(graph):
        subg = graph.subgraph(nodes)
        if subg.number_of_nodes() <= 1:
            diameters.append(0.0)
        else:
            diameters.append(float(nx.diameter(subg)))
    return float(max(diameters, default=0.0))


def encode_subgraph(graph: nx.Graph) -> dict[str, float]:
    num_nodes = float(graph.number_of_nodes())
    num_edges = float(graph.number_of_edges())

    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0.0,
            "num_edges": 0.0,
            "avg_degree": 0.0,
            "max_degree": 0.0,
            "var_degree": 0.0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "triangles": 0.0,
            "diameter": 0.0,
        }

    degrees = np.array([d for _, d in graph.degree()], dtype=np.float64)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": float(degrees.mean()),
        "max_degree": float(degrees.max(initial=0.0)),
        "var_degree": float(degrees.var()),
        "density": float(nx.density(graph)),
        "avg_clustering": float(nx.average_clustering(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "triangles": _safe_triangle_count(graph),
        "diameter": _safe_diameter(graph),
    }


def build_stage1_records(
    dataset_name: str,
    data_root: str,
    max_subgraph_nodes: int,
    remove_selfloops: bool,
    subgraph_hop_n: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    sample_id = 0
    for graph_id, nx_graph in iter_dataset_graphs(dataset_name, data_root, remove_selfloops):
        subgraphs = split_graph_to_subgraphs(
            nx_graph,
            max_nodes=max_subgraph_nodes,
            hop_n=subgraph_hop_n,
        )

        for subgraph_id, subg in enumerate(subgraphs):
            encoded = encode_subgraph(subg)
            record: dict[str, float | int] = {
                "sample_id": sample_id,
                "graph_id": graph_id,
                "subgraph_id": subgraph_id,
            }
            record.update(encoded)
            records.append(record)
            sample_id += 1

    return records


def run_stage1(config: Stage1Config) -> dict[str, str | int]:
    records = build_stage1_records(
        dataset_name=config.dataset,
        data_root=config.data_root,
        max_subgraph_nodes=config.max_subgraph_nodes,
        remove_selfloops=config.remove_selfloops,
        subgraph_hop_n=config.subgraph_hop_n,
    )

    labels, centers_raw, feature_means, x_scaled, centers_scaled, cluster_feature_names = fit_clusters(
        records=records,
        num_clusters=config.num_clusters,
        seed=config.seed,
        use_log1p=config.use_log1p,
        encoding_mode=config.encoding_mode,
        cluster_method=config.cluster_method,
    )

    write_info = write_stage1_outputs(
        output_root=config.output_root,
        dataset=config.dataset,
        records=records,
        labels=labels,
        centers_raw=centers_raw,
        x_scaled=x_scaled,
        centers_scaled=centers_scaled,
        max_subgraph_nodes=config.max_subgraph_nodes,
        subgraph_hop_n=config.subgraph_hop_n,
        num_clusters=config.num_clusters,
        remove_selfloops=config.remove_selfloops,
        encoding_mode=config.encoding_mode,
        cluster_method=config.cluster_method,
        use_log1p=config.use_log1p,
        cluster_feature_names=cluster_feature_names,
        feature_means=feature_means,
    )

    return {
        "dataset": config.dataset,
        "num_samples": len(records),
        "effective_k": int(write_info["effective_k"]),
        "output_dir": str(write_info["output_dir"]),
        "summary_json": str(write_info["summary_json"]),
    }


def run_stage2(config: Stage2Config) -> dict[str, str | int | float]:
    csv_path = Path(config.input_csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"input csv not found: {csv_path}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(csv_path)
    x = np.column_stack([table[k].astype(float) for k in FEATURE_KEYS]).astype(np.float64)
    sample_ids = table["sample_id"].astype(int)
    graph_ids = table["graph_id"].astype(int) if "graph_id" in table else np.zeros(len(sample_ids), dtype=np.int64)
    subgraph_ids = table["subgraph_id"].astype(int) if "subgraph_id" in table else np.arange(len(sample_ids), dtype=np.int64)
    cluster_ids = table["cluster_id"].astype(int)

    centers_raw = None
    if config.center_path:
        center_path = Path(config.center_path).resolve()
    else:
        center_path = csv_path.parent / "cluster_centers.npy"
    if center_path.exists():
        centers_raw = np.load(center_path)

    rep_indices_stage1, rep_dist_stage1 = build_representative_pool(
        x=x,
        cluster_ids=cluster_ids,
        top_m_per_cluster=config.top_m_per_cluster,
        centers_raw=centers_raw,
    )
    novelty, coverage, redundancy, novelty_n, coverage_n, redundancy_n, learnability, order = compute_learnability(
        x=x,
        cluster_ids=cluster_ids,
        rep_indices=rep_indices_stage1,
        alpha=float(config.alpha),
        beta=float(config.beta),
        gamma=float(config.gamma),
    )

    # Stage-2 second selection: top-p% by learnability from remaining pool.
    all_idx = np.arange(len(sample_ids), dtype=np.int64)
    stage1_set = set(int(i) for i in rep_indices_stage1.tolist())
    remain_idx = np.array([i for i in all_idx.tolist() if int(i) not in stage1_set], dtype=np.int64)
    top_p = float(np.clip(config.top_p_percent, 0.0, 1.0))
    second_take = int(np.ceil(remain_idx.size * top_p)) if remain_idx.size > 0 else 0
    if remain_idx.size > 0 and second_take > 0:
        remain_order = remain_idx[np.argsort(-learnability[remain_idx])]
        rep_indices_stage2 = remain_order[:second_take]
    else:
        rep_indices_stage2 = np.array([], dtype=np.int64)

    rep_indices = np.concatenate([rep_indices_stage1, rep_indices_stage2], axis=0)
    if rep_indices.size > 0:
        rep_indices = np.unique(rep_indices)

    # Keep distance-to-center for stage1 picks only; stage2 picks set to -1.
    rep_dist_map = {int(i): float(d) for i, d in zip(rep_indices_stage1.tolist(), rep_dist_stage1.tolist())}
    rep_dist = np.array([rep_dist_map.get(int(i), -1.0) for i in rep_indices.tolist()], dtype=np.float64)

    in_pool = np.zeros(len(sample_ids), dtype=np.int64)
    if rep_indices.size > 0:
        in_pool[rep_indices] = 1

    def _cluster_ratio(indices: np.ndarray) -> dict[str, float]:
        if indices.size == 0:
            return {}
        ids = cluster_ids[indices]
        uniq, cnt = np.unique(ids, return_counts=True)
        total = float(max(1, cnt.sum()))
        return {str(int(c)): float(n / total) for c, n in zip(uniq.tolist(), cnt.tolist())}

    stage_debug = {
        "num_samples_total": int(len(sample_ids)),
        "stage1_top_m_per_cluster": int(config.top_m_per_cluster),
        "stage2_top_p_percent": float(top_p),
        "stage1_selected_count": int(rep_indices_stage1.size),
        "stage2_selected_count": int(rep_indices_stage2.size),
        "selected_total_count": int(rep_indices.size),
        "stage1_cluster_ratio": _cluster_ratio(rep_indices_stage1),
        "stage2_cluster_ratio": _cluster_ratio(rep_indices_stage2),
        "selected_total_cluster_ratio": _cluster_ratio(rep_indices),
    }

    # Decode representative pool back to subgraphs and export per-sample matrices.
    decoded_summary = {
        "enabled": bool(config.write_decoded_subgraphs),
        "decoded_count": 0,
        "decoded_root": "",
        "dataset_prompt_file": "",
    }
    if config.write_decoded_subgraphs and config.dataset_name:
        try:
            data_root = config.data_root if config.data_root else str((PROJECT_ROOT / "data").resolve())
            decoded_root = output_dir / "decoded_subgraphs"
            decoded_root.mkdir(parents=True, exist_ok=True)

            # Same dataset shares one prompt vector.
            rng = np.random.default_rng(2026 + sum(ord(ch) for ch in config.dataset_name.lower()))
            prompt_vec = rng.normal(0.0, 0.02, size=(int(config.pca_dim),)).astype(np.float32)
            prompt_path = output_dir / "dataset_prompt.npy"
            np.save(prompt_path, prompt_vec)

            try:
                ds_obj = load_dataset(config.dataset_name, path=data_root)
            except Exception:
                ds_obj = load_dataset(config.dataset_name.capitalize(), path=data_root)

            graph_cache: dict[int, tuple[np.ndarray, np.ndarray, list[nx.Graph]]] = {}
            decoded_ok = 0

            for pick_idx in rep_indices.tolist():
                g_id = int(graph_ids[pick_idx])
                sg_id = int(subgraph_ids[pick_idx])

                if g_id not in graph_cache:
                    data = ds_obj[g_id]
                    feat_raw, adj_raw = process.process_tu(data, data.x.shape[1])
                    feat_raw = np.asarray(feat_raw, dtype=np.float32)

                    eff_dim = int(min(config.pca_dim, feat_raw.shape[1], feat_raw.shape[0]))
                    if eff_dim > 0:
                        from sklearn.decomposition import PCA

                        pca = PCA(n_components=eff_dim)
                        feat_pca_full = pca.fit_transform(feat_raw).astype(np.float32)
                        if eff_dim < int(config.pca_dim):
                            pad = np.zeros((feat_pca_full.shape[0], int(config.pca_dim) - eff_dim), dtype=np.float32)
                            feat_pca_full = np.concatenate([feat_pca_full, pad], axis=1)
                    else:
                        feat_pca_full = np.zeros((feat_raw.shape[0], int(config.pca_dim)), dtype=np.float32)

                    g_ref = get_stage1_reference_graph(
                        dataset_name=config.dataset_name,
                        data_root=data_root,
                        graph_id=g_id,
                        remove_selfloops=bool(config.remove_selfloops),
                    )
                    subgraphs = split_graph_to_subgraphs(
                        g_ref,
                        max_nodes=int(config.max_subgraph_nodes),
                        hop_n=int(config.subgraph_hop_n),
                    )
                    graph_cache[g_id] = (feat_pca_full, adj_raw, subgraphs)

                feat_pca_full, adj_raw, subgraphs = graph_cache[g_id]
                if sg_id < 0 or sg_id >= len(subgraphs):
                    continue

                node_idx = np.array(sorted(list(subgraphs[sg_id].nodes())), dtype=np.int64)
                if node_idx.size == 0:
                    continue

                feat_pca = np.asarray(feat_pca_full[node_idx, :], dtype=np.float32)
                sub_adj = np.asarray(adj_raw[node_idx][:, node_idx].todense(), dtype=np.float32)
                np.fill_diagonal(sub_adj, 0.0)
                sub_adj_norm = process.normalize_adj(sp.csr_matrix(sub_adj) + sp.eye(sub_adj.shape[0]))
                sub_adj_norm = np.asarray(sub_adj_norm.todense(), dtype=np.float32)

                feat_prompted = feat_pca + prompt_vec[None, :]

                np.savez_compressed(
                    decoded_root / f"sample_{int(sample_ids[pick_idx])}.npz",
                    sample_id=np.int64(sample_ids[pick_idx]),
                    graph_id=np.int64(g_id),
                    subgraph_id=np.int64(sg_id),
                    cluster_id=np.int64(cluster_ids[pick_idx]),
                    node_idx=node_idx,
                    feature_matrix=feat_pca,
                    feature_matrix_prompted=feat_prompted,
                    adjacency_matrix=sub_adj,
                    adjacency_matrix_normalized=sub_adj_norm,
                )
                decoded_ok += 1

            decoded_summary = {
                "enabled": True,
                "decoded_count": int(decoded_ok),
                "decoded_root": str(decoded_root),
                "dataset_prompt_file": str(prompt_path),
            }
        except Exception as e:
            decoded_summary = {
                "enabled": False,
                "decoded_count": 0,
                "decoded_root": "",
                "dataset_prompt_file": "",
                "error": str(e),
            }

    print(
        "[stage2-debug] "
        f"stage1_selected={stage_debug['stage1_selected_count']} "
        f"stage2_selected={stage_debug['stage2_selected_count']} "
        f"selected_total={stage_debug['selected_total_count']}"
    )
    print(f"[stage2-debug] stage1_cluster_ratio={stage_debug['stage1_cluster_ratio']}")
    print(f"[stage2-debug] stage2_cluster_ratio={stage_debug['stage2_cluster_ratio']}")
    print(f"[stage2-debug] selected_total_cluster_ratio={stage_debug['selected_total_cluster_ratio']}")

    write_info = write_stage2_outputs(
        output_dir=output_dir,
        input_csv=csv_path,
        sample_ids=sample_ids,
        graph_ids=graph_ids,
        subgraph_ids=subgraph_ids,
        cluster_ids=cluster_ids,
        rep_indices=rep_indices,
        rep_dist=rep_dist,
        in_pool=in_pool,
        novelty=novelty,
        coverage=coverage,
        redundancy=redundancy,
        novelty_n=novelty_n,
        coverage_n=coverage_n,
        redundancy_n=redundancy_n,
        learnability=learnability,
        order=order,
        top_m_per_cluster=int(config.top_m_per_cluster),
        top_p_percent=float(top_p),
        alpha=float(config.alpha),
        beta=float(config.beta),
        gamma=float(config.gamma),
        stage_debug=stage_debug,
        decoded_summary=decoded_summary,
    )

    return {
        "num_samples": int(len(sample_ids)),
        "representative_pool_size": int(write_info["representative_pool_size"]),
        "summary_json": str(write_info["summary_json"]),
        "scores_csv": str(write_info["scores_csv"]),
    }


__all__ = [
    "Stage1Config",
    "Stage2Config",
    "build_stage1_records",
    "encode_subgraph",
    "get_stage1_reference_graph",
    "run_stage1",
    "run_stage2",
    "split_graph_to_subgraphs",
]
