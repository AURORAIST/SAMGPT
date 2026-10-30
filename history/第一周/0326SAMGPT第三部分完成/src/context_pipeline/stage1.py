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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.in_memory_dataset")
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
    use_log1p: bool = True
    subgraph_hop_n: int = 2
    encoding_mode: str = "raw9"


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


def get_stage1_reference_graph(
    dataset_name: str,
    data_root: str,
    graph_id: int,
    remove_selfloops: bool,
) -> nx.Graph:
    """Return the exact graph variant used by stage1 for a given graph_id.

    This keeps downstream subgraph reconstruction consistent with stage1 outputs
    (especially for Planetoid datasets where stage1 may use raw files).
    """
    for gid, g in _iter_dataset_graphs(dataset_name, data_root, remove_selfloops):
        if int(gid) == int(graph_id):
            return g
    raise IndexError(
        f"graph_id={graph_id} not found for dataset={dataset_name} under data_root={data_root}"
    )


def _bfs_chunk(component_graph: nx.Graph, start_node: int, max_nodes: int, unassigned: set[int]) -> list[int]:
    queue: deque[int] = deque([start_node])
    queued = {start_node}
    visited: set[int] = set()
    chunk: list[int] = []

    while queue and len(chunk) < max_nodes:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        # Allow BFS to traverse through already-assigned nodes so later chunks
        # are not artificially fragmented by earlier cuts.
        if node in unassigned:
            unassigned.remove(node)
            chunk.append(node)

        # Keep neighbor traversal deterministic so subgraph_id is stable across runs.
        for nbr in sorted(component_graph.neighbors(node)):
            if nbr not in queued:
                queued.add(nbr)
                queue.append(nbr)

    return chunk


def _n_hop_seed_chunk(
    component_graph: nx.Graph,
    start_node: int,
    max_nodes: int,
    hop_n: int,
) -> list[int]:
    """Build a per-center n-hop neighborhood chunk (baseline-aligned semantics).

    Unlike partition-based slicing, this function does not consume a global
    unassigned set: nodes can appear in multiple center-based subgraphs.
    """
    chunk: list[int] = []
    selected: set[int] = set()

    def _take(node: int) -> None:
        if len(chunk) >= max_nodes:
            return
        if node not in selected:
            selected.add(node)
            chunk.append(node)

    _take(start_node)

    frontier: set[int] = {start_node}
    for _ in range(max(0, int(hop_n))):
        if len(chunk) >= max_nodes or not frontier:
            break

        candidates: set[int] = set()
        for node in sorted(frontier):
            candidates.update(component_graph.neighbors(node))

        next_frontier: set[int] = set()
        for cand in sorted(candidates):
            if len(chunk) >= max_nodes:
                break
            _take(cand)
            next_frontier.add(cand)

        frontier = next_frontier

    return chunk


def _bfs_complete_chunk(
    component_graph: nx.Graph,
    seed_nodes: list[int],
    max_nodes: int,
    unassigned: set[int],
) -> list[int]:
    """Complete a chunk via BFS while allowing pass-through on assigned nodes."""
    queue: deque[int] = deque(seed_nodes)
    queued = set(seed_nodes)
    visited: set[int] = set()
    chunk: list[int] = []

    while queue and len(chunk) < max_nodes:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node in unassigned:
            unassigned.remove(node)
            chunk.append(node)

        for nbr in sorted(component_graph.neighbors(node)):
            if nbr not in queued:
                queued.add(nbr)
                queue.append(nbr)

    return chunk


def split_graph_to_subgraphs(graph: nx.Graph, max_nodes: int, hop_n: int = 2) -> list[nx.Graph]:
    if graph.number_of_nodes() == 0:
        return []

    subgraphs: list[nx.Graph] = []
    components = list(nx.connected_components(graph))
    components.sort(key=lambda nodes: min(nodes) if nodes else -1)

    for component_nodes in components:
        comp = graph.subgraph(component_nodes).copy()
        # Baseline-like center sampling: one subgraph per center node.
        for seed in sorted(comp.nodes()):
            chunk_nodes = _n_hop_seed_chunk(
                comp,
                seed,
                max_nodes=max_nodes,
                hop_n=hop_n,
            )
            if not chunk_nodes:
                chunk_nodes = [seed]
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
    subgraph_hop_n: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    sample_id = 0
    for graph_id, nx_graph in _iter_dataset_graphs(dataset_name, data_root, remove_selfloops):
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


RAW_FEATURE_NAMES = [
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


def _build_cluster_input(
    x_raw: np.ndarray,
    use_log1p: bool,
    encoding_mode: str,
) -> tuple[np.ndarray, list[str]]:
    mode = encoding_mode.strip().lower()
    if mode == "raw9":
        x_trans = _apply_log1p_selected(x_raw) if use_log1p else x_raw.copy()
        return x_trans, RAW_FEATURE_NAMES

    if mode != "decorrelated7":
        raise ValueError(f"Unsupported encoding_mode={encoding_mode}. Use 'raw9' or 'decorrelated7'.")

    num_nodes = np.maximum(x_raw[:, 0], 0.0)
    num_edges = np.maximum(x_raw[:, 1], 0.0)
    avg_degree = np.maximum(x_raw[:, 2], 0.0)
    max_degree = np.maximum(x_raw[:, 3], 0.0)
    var_degree = np.maximum(x_raw[:, 4], 0.0)
    density = np.clip(x_raw[:, 5], 0.0, 1.0)
    avg_clustering = np.clip(x_raw[:, 6], 0.0, 1.0)
    triangles = np.maximum(x_raw[:, 7], 0.0)
    diameter = np.maximum(x_raw[:, 8], 0.0)

    log_num_edges = np.log1p(num_edges)
    hub_ratio = max_degree / np.maximum(num_nodes - 1.0, 1.0)
    degree_cv = np.sqrt(var_degree) / np.maximum(avg_degree, 1e-6)
    tri_denom = np.maximum(num_nodes * np.maximum(num_nodes - 1.0, 0.0) * np.maximum(num_nodes - 2.0, 0.0) / 6.0, 1.0)
    triangle_rate = triangles / tri_denom
    diameter_norm = diameter / np.maximum(num_nodes - 1.0, 1.0)

    x_enc = np.column_stack([
        log_num_edges,
        hub_ratio,
        degree_cv,
        density,
        avg_clustering,
        triangle_rate,
        diameter_norm,
    ]).astype(np.float64)
    names = [
        "log_num_edges",
        "hub_ratio",
        "degree_cv",
        "density",
        "avg_clustering",
        "triangle_rate",
        "diameter_norm",
    ]
    return x_enc, names


def _cluster_centers_in_raw9(x_raw: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, x_raw.shape[1]), dtype=np.float64)
    global_mean = x_raw.mean(axis=0)
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            centers[cid] = global_mean
        else:
            centers[cid] = x_raw[idx].mean(axis=0)
    return centers


LOG1P_FEATURE_IDX = [0, 1, 3, 4, 7, 8]  # num_nodes,num_edges,max_degree,var_degree,triangles,diameter


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
    use_log1p: bool,
    encoding_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not records:
        raise ValueError("No records to cluster.")

    x_raw = _matrix_from_records(records)
    x_cluster, cluster_feature_names = _build_cluster_input(
        x_raw=x_raw,
        use_log1p=use_log1p,
        encoding_mode=encoding_mode,
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_cluster)

    k = min(max(1, num_clusters), x_scaled.shape[0])
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, reg_covar=1e-6)
    labels = model.fit_predict(x_scaled)
    centers_scaled = model.means_

    # Keep stage2 compatibility: always provide raw-9 cluster centers.
    centers_raw9 = _cluster_centers_in_raw9(x_raw=x_raw, labels=labels, k=k)

    return labels, centers_raw9, scaler.mean_, x_scaled, centers_scaled, cluster_feature_names


def _save_cluster_visualization(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    output_path: Path,
    dataset_name: str,
    remove_selfloops: bool,
) -> None:
    """Save PCA plot for clustered samples and highlight cluster centers."""
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
        subgraph_hop_n=config.subgraph_hop_n,
    )

    labels, centers_raw, feature_means, x_scaled, centers_scaled, cluster_feature_names = fit_clusters(
        records=records,
        num_clusters=config.num_clusters,
        seed=config.seed,
        use_log1p=config.use_log1p,
        encoding_mode=config.encoding_mode,
    )

    output_dir = Path(config.output_root) / config.dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "structural_encodings.csv"
    npy_labels = output_dir / "cluster_labels.npy"
    npy_centers = output_dir / "cluster_centers.npy"
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
        "subgraph_hop_n": int(config.subgraph_hop_n),
        "cluster_method": "gmm",
        "encoding_mode": config.encoding_mode,
        "cluster_feature_names": cluster_feature_names,
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
