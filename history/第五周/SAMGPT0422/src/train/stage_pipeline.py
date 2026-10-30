from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

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
from models.stage1_gmm_cluster import fit_clusters
from models.stage2_learnability_model import FEATURE_KEYS, compute_learnability  # type: ignore[reportMissingImports]


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
    feature_k: int = 0
    feature_mask: str = "123456789"


@dataclass
class Stage2Config:
    input_csv: str
    output_dir: str
    top_m_per_cluster: int = 5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    center_path: str = ""


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

    labels, centers_raw, feature_means, x_scaled, centers_scaled, cluster_feature_names, scaler_mean_out, scaler_scale_out = fit_clusters(
        records=records,
        num_clusters=config.num_clusters,
        seed=config.seed,
        use_log1p=config.use_log1p,
        encoding_mode=config.encoding_mode,
        feature_k=int(config.feature_k),
        feature_mask=str(config.feature_mask),
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
        feature_k=int(config.feature_k),
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

    rep_indices, rep_dist = build_representative_pool(
        x=x,
        cluster_ids=cluster_ids,
        top_m_per_cluster=config.top_m_per_cluster,
        centers_raw=centers_raw,
    )
    novelty, coverage, redundancy, novelty_n, coverage_n, redundancy_n, learnability, order = compute_learnability(
        x=x,
        cluster_ids=cluster_ids,
        rep_indices=rep_indices,
        alpha=float(config.alpha),
        beta=float(config.beta),
        gamma=float(config.gamma),
    )

    in_pool = np.zeros(len(sample_ids), dtype=np.int64)
    if rep_indices.size > 0:
        in_pool[rep_indices] = 1
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
        alpha=float(config.alpha),
        beta=float(config.beta),
        gamma=float(config.gamma),
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
