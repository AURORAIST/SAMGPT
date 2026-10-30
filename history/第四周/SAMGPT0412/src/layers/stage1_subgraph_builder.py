from __future__ import annotations

import pickle as pkl
import sys
import warnings
from pathlib import Path

import networkx as nx
from torch_geometric.utils import to_networkx

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.in_memory_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.dataset import load_dataset  # noqa: E402  # type: ignore[reportMissingImports]

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


def iter_dataset_graphs(dataset_name: str, data_root: str, remove_selfloops: bool):
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
    for gid, g in iter_dataset_graphs(dataset_name, data_root, remove_selfloops):
        if int(gid) == int(graph_id):
            return g
    raise IndexError(
        f"graph_id={graph_id} not found for dataset={dataset_name} under data_root={data_root}"
    )


def _n_hop_seed_chunk(
    component_graph: nx.Graph,
    start_node: int,
    max_nodes: int,
    hop_n: int,
) -> list[int]:
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


def split_graph_to_subgraphs(graph: nx.Graph, max_nodes: int, hop_n: int = 2) -> list[nx.Graph]:
    if graph.number_of_nodes() == 0:
        return []

    subgraphs: list[nx.Graph] = []
    components = list(nx.connected_components(graph))
    components.sort(key=lambda nodes: min(nodes) if nodes else -1)

    for component_nodes in components:
        comp = graph.subgraph(component_nodes).copy()
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
