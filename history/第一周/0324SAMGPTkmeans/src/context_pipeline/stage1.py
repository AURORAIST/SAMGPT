from __future__ import annotations

import json
import pickle as pkl
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx

# Reuse SAMGPT dataset loader.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.dataset import load_dataset  # noqa: E402  # type: ignore[reportMissingImports]


@dataclass
class Stage1Config:
    dataset: str
    data_root: str
    output_root: str
    max_subgraph_nodes: int = 64
    num_clusters: int = 8
    seed: int = 0
    remove_selfloops: bool = True
    cluster_method: str = "gmm"
    use_log1p: bool = True


PLANETOID_DATASETS = {"Cora", "Citeseer", "Pubmed"}


def _parse_index_file(index_path: Path) -> list[int]:
    indices: list[int] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))
    return indices


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pkl.load(f, encoding="latin1")


def _load_planetoid_graph_from_raw(dataset_name: str, data_root: str) -> nx.Graph:
    ds_lower = dataset_name.lower()
    raw_dir = Path(data_root) / dataset_name / "raw"

    graph_path = raw_dir / f"ind.{ds_lower}.graph"
    allx_path = raw_dir / f"ind.{ds_lower}.allx"
    tx_path = raw_dir / f"ind.{ds_lower}.tx"
    test_index_path = raw_dir / f"ind.{ds_lower}.test.index"

    required_paths = [graph_path, allx_path, tx_path, test_index_path]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Planetoid raw files: {missing}")

    graph_dict = _load_pickle(graph_path)
    allx = _load_pickle(allx_path)
    tx = _load_pickle(tx_path)
    test_idx = _parse_index_file(test_index_path)

    g = nx.from_dict_of_lists(graph_dict).to_undirected()

    max_graph_node = max(g.nodes(), default=-1)
    max_test_idx = max(test_idx) if test_idx else -1
    num_from_features = int(allx.shape[0] + tx.shape[0])
    num_nodes = max(max_graph_node + 1, max_test_idx + 1, num_from_features)
    if num_nodes > 0:
        g.add_nodes_from(range(num_nodes))

    return g


def _iter_dataset_graphs(dataset_name: str, data_root: str, remove_selfloops: bool):
    if dataset_name in PLANETOID_DATASETS:
        try:
            g = _load_planetoid_graph_from_raw(dataset_name, data_root)
            if remove_selfloops:
                g.remove_edges_from(nx.selfloop_edges(g))
            yield 0, g
            return
        except Exception:
            pass

    dataset = load_dataset(dataset_name, path=data_root)
    for graph_id, data in enumerate(dataset):
        g = to_networkx(data, to_undirected=True)
        if remove_selfloops:
            g.remove_edges_from(nx.selfloop_edges(g))
        yield graph_id, g


def _bfs_chunk(component_graph: nx.Graph, start_node: int, max_nodes: int, unassigned: set[int]) -> list[int]:
    queue: deque[int] = deque([start_node])
    queued = {start_node}
    chunk: list[int] = []

    while queue and len(chunk) < max_nodes:
        node = queue.popleft()
        if node not in unassigned:
            continue
        unassigned.remove(node)
        chunk.append(node)

        for nbr in component_graph.neighbors(node):
            if nbr in unassigned and nbr not in queued:
                queued.add(nbr)
                queue.append(nbr)

    return chunk


def split_graph_to_subgraphs(graph: nx.Graph, max_nodes: int) -> list[nx.Graph]:
    if graph.number_of_nodes() <= max_nodes:
        return [graph.copy()]

    subgraphs: list[nx.Graph] = []
    for component_nodes in nx.connected_components(graph):
        comp = graph.subgraph(component_nodes).copy()
        unassigned: set[int] = set(comp.nodes())

        while unassigned:
            seed = next(iter(unassigned))
            chunk_nodes = _bfs_chunk(comp, seed, max_nodes=max_nodes, unassigned=unassigned)
            subgraphs.append(comp.subgraph(chunk_nodes).copy())

    return subgraphs


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
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    sample_id = 0
    for graph_id, nx_graph in _iter_dataset_graphs(dataset_name, data_root, remove_selfloops):
        subgraphs = split_graph_to_subgraphs(nx_graph, max_nodes=max_subgraph_nodes)

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


def _matrix_from_records(records: Iterable[dict[str, float | int]]) -> np.ndarray:
    keys = [
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
    return np.array([[float(r[k]) for k in keys] for r in records], dtype=np.float64)


LOG1P_FEATURE_IDX = [0, 1, 3, 4, 7, 8]


def _apply_log1p_selected(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for idx in LOG1P_FEATURE_IDX:
        out[:, idx] = np.log1p(np.maximum(out[:, idx], 0.0))
    return out


def _inverse_log1p_selected(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for idx in LOG1P_FEATURE_IDX:
        out[:, idx] = np.expm1(out[:, idx])
    return out


def fit_clusters(
    records: list[dict[str, float | int]],
    num_clusters: int,
    seed: int,
    cluster_method: str,
    use_log1p: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No records to cluster.")

    x = _matrix_from_records(records)
    x_trans = _apply_log1p_selected(x) if use_log1p else x.copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_trans)

    k = min(max(1, num_clusters), x_scaled.shape[0])
    method = cluster_method.lower().strip()
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(x_scaled)
        centers_scaled = model.cluster_centers_
    elif method == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, reg_covar=1e-6)
        labels = model.fit_predict(x_scaled)
        centers_scaled = model.means_
    else:
        raise ValueError(f"Unsupported cluster_method: {cluster_method}. Use 'gmm' or 'kmeans'.")

    centers_trans = scaler.inverse_transform(centers_scaled)
    centers_raw = _inverse_log1p_selected(centers_trans) if use_log1p else centers_trans

    return labels, centers_raw, scaler.mean_, x_scaled, centers_scaled


def _save_cluster_visualization(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    output_path: Path,
    dataset_name: str,
    remove_selfloops: bool,
) -> None:
    pca = PCA(n_components=2, random_state=0)
    x2d = pca.fit_transform(x_scaled)
    c2d = pca.transform(centers_scaled)

    unique_clusters, counts = np.unique(labels, return_counts=True)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    scatter = ax1.scatter(
        x2d[:, 0],
        x2d[:, 1],
        c=labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
        edgecolors="none",
    )
    ax1.scatter(
        c2d[:, 0],
        c2d[:, 1],
        marker="o",
        s=90,
        c="red",
        edgecolors="none",
        zorder=4,
        label="Cluster Centers",
    )

    ax1.set_title("Stage-1 Cluster Visualization (PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label("cluster_id")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(unique_clusters, counts, color="#2f6c8f")
    ax2.set_title("Cluster Size Distribution")
    ax2.set_xlabel("cluster_id")
    ax2.set_ylabel("count")
    ax2.set_xticks(unique_clusters)

    fig.suptitle(
        f"{dataset_name} | remove_selfloops={int(remove_selfloops)} | clusters={len(unique_clusters)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_stage1(config: Stage1Config) -> dict[str, str | int]:
    records = build_stage1_records(
        dataset_name=config.dataset,
        data_root=config.data_root,
        max_subgraph_nodes=config.max_subgraph_nodes,
        remove_selfloops=config.remove_selfloops,
    )

    labels, centers_raw, feature_means, x_scaled, centers_scaled = fit_clusters(
        records=records,
        num_clusters=config.num_clusters,
        seed=config.seed,
        cluster_method=config.cluster_method,
        use_log1p=config.use_log1p,
    )

    output_dir = Path(config.output_root) / config.dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "structural_encodings.csv"
    npy_labels = output_dir / "cluster_labels.npy"
    npy_centers = output_dir / "cluster_centers.npy"
    legacy_npy_labels = output_dir / "kmeans_labels.npy"
    legacy_npy_centers = output_dir / "kmeans_centers.npy"
    viz_path = output_dir / "stage1_cluster_visualization.png"
    summary_json = output_dir / "stage1_summary.json"

    header = [
        "sample_id",
        "graph_id",
        "subgraph_id",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "max_degree",
        "var_degree",
        "density",
        "avg_clustering",
        "triangles",
        "diameter",
        "cluster_id",
    ]

    lines = [",".join(header)]
    for rec, cluster in zip(records, labels.tolist()):
        row = [
            str(rec["sample_id"]),
            str(rec["graph_id"]),
            str(rec["subgraph_id"]),
            str(rec["num_nodes"]),
            str(rec["num_edges"]),
            str(rec["avg_degree"]),
            str(rec["max_degree"]),
            str(rec["var_degree"]),
            str(rec["density"]),
            str(rec["avg_clustering"]),
            str(rec["triangles"]),
            str(rec["diameter"]),
            str(cluster),
        ]
        lines.append(",".join(row))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(npy_labels, labels)
    np.save(npy_centers, centers_raw)
    np.save(legacy_npy_labels, labels)
    np.save(legacy_npy_centers, centers_raw)

    _save_cluster_visualization(
        x_scaled=x_scaled,
        labels=labels,
        centers_scaled=centers_scaled,
        output_path=viz_path,
        dataset_name=config.dataset,
        remove_selfloops=config.remove_selfloops,
    )

    summary = {
        "dataset": config.dataset,
        "num_samples": len(records),
        "max_subgraph_nodes": config.max_subgraph_nodes,
        "cluster_method": config.cluster_method,
        "use_log1p": bool(config.use_log1p),
        "log1p_feature_names": [
            "num_nodes",
            "num_edges",
            "max_degree",
            "var_degree",
            "triangles",
            "diameter",
        ],
        "requested_num_clusters": config.num_clusters,
        "effective_num_clusters": int(centers_raw.shape[0]),
        "remove_selfloops": bool(config.remove_selfloops),
        "feature_means": feature_means.tolist(),
        "paths": {
            "encodings_csv": str(csv_path),
            "cluster_labels": str(npy_labels),
            "cluster_centers": str(npy_centers),
            "legacy_kmeans_labels": str(legacy_npy_labels),
            "legacy_kmeans_centers": str(legacy_npy_centers),
            "cluster_visualization": str(viz_path),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset": config.dataset,
        "num_samples": len(records),
        "effective_k": int(centers_raw.shape[0]),
        "output_dir": str(output_dir),
        "summary_json": str(summary_json),
    }
