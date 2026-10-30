from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from models.stage2_learnability_model import compute_learnability
from tools.encoder_filter_diagnostics import assess_cluster_quality, compute_cluster_quality_metrics, save_encoder_filter_visualization

from downprompt import downprompt, downprompt_graph
from preprompt import PrePrompt, pca_compression
import preprompt
from utils.dataset import load_dataset
from utils import aug
from utils import process

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEBUG_TRACE_ENABLE = True


def torch_load_trusted(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_trusted(path: str, map_location=None):
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd


def debug_trace(event, **kwargs):
    if not DEBUG_TRACE_ENABLE:
        return
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if kwargs:
        details = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        print(f'[trace] {ts} {event}: {details}')
    else:
        print(f'[trace] {ts} {event}')


parser = argparse.ArgumentParser("SAMGPT")

# Basic settings
parser.add_argument('--dataset', type=str, default="Cora", help='downstream dataset')
parser.add_argument(
    '--pretrain_datasets',
    nargs='+',
    type=str,
    default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'],
    help='pretrain datasets',
)
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--pretrain_method', type=str, default='EGRAPHCL', choices=['EGRAPHCL'], help='pretrain method (fixed: EGRAPHCL)')

# Training/data
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='target graph id (1-based)')
parser.add_argument('--skip_pretrain', type=int, default=0, help='1: load checkpoint and skip pretrain')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hidden dim')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='gcn or gat')
parser.add_argument('--unify_dim', type=int, default=50, help='input dim after PCA')
parser.add_argument('--feature_align_method', type=str, default='pca', choices=['svd', 'pca'], help='cross-domain feature alignment method')
parser.add_argument('--alpha', type=float, default=1.0, help='combine alpha in preprompt')
parser.add_argument('--beta', type=float, default=1.0, help='beta in downstream combine')
parser.add_argument('--combinetype', type=str, default='mul', help='combine type')
parser.add_argument('--ablation_pre', type=str, default='st', choices=['st'], help='pretrain prompt mode (fixed: st)')
parser.add_argument('--ablation_down', type=str, default='all', help='all/ft/st/None')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative sample num for LP')

# Optim controls
parser.add_argument('--nb_epochs', type=int, default=10000, help='pretrain max epochs')
parser.add_argument('--patience', type=int, default=50, help='early stop patience')
parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='grad clip; <=0 disable')
parser.add_argument('--lr_scheduler_patience', type=int, default=20, help='ReduceLROnPlateau patience')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='ReduceLROnPlateau factor')
parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-5, help='ReduceLROnPlateau min lr')
parser.add_argument('--early_stop_min_delta', type=float, default=1e-4, help='minimum loss improvement')
parser.add_argument('--epoch_log_every', type=int, default=5, help='epoch log interval')

# EGRAPHCL options
parser.add_argument('--egraphcl_tau', type=float, default=0.85, help='online soft positive threshold')
parser.add_argument('--egraphcl_temperature', type=float, default=0.2, help='temperature')
parser.add_argument('--egraphcl_pair_chunk_size', type=int, default=256, help='pair chunk')
parser.add_argument('--egraphcl_candidate_pool', type=int, default=1024, help='candidate pool')
parser.add_argument('--egraphcl_soft_topk', type=int, default=10, help='soft topk')
parser.add_argument('--egraphcl_max_negatives', type=int, default=512, help='max negatives')
parser.add_argument('--egraphcl_batch_size', type=int, default=128, help='egraphcl minibatch size')
parser.add_argument('--egraphcl_keep_data_on_cpu', type=int, default=1, help='1: keep full EGRAPHCL sample pool on CPU and stream by minibatch')
parser.add_argument('--egraphcl_epoch_chunk_graphs', type=int, default=2048, help='outer chunk size (graphs) for per-epoch backward; <=0 disables')
parser.add_argument('--egraphcl_mask_source', type=str, default='feature', choices=['structure', 'feature'], help='legacy arg')
parser.add_argument('--egraphcl_adaptive_tau', type=int, default=1, help='1: use batch-quantile adaptive tau for soft mask')
parser.add_argument('--egraphcl_target_mask_ratio', type=float, default=0.35, help='target masked-pair ratio for adaptive tau')
parser.add_argument('--egraphcl_tau_min', type=float, default=0.80, help='lower bound of adaptive tau')
parser.add_argument('--egraphcl_tau_max', type=float, default=0.99, help='upper bound of adaptive tau')
parser.add_argument('--egraphcl_min_negatives_per_anchor', type=int, default=1, help='min negatives per anchor')
parser.add_argument('--egraphcl_structure_min_negatives_per_anchor', type=int, default=8, help='legacy arg')
parser.add_argument('--egraphcl_max_mask_ratio', type=float, default=0.95, help='max masked ratio')
parser.add_argument('--egraphcl_mask_log_every', type=int, default=5, help='mask log interval')

# New encode->cluster->filter controls (replace stage1/stage2)
parser.add_argument('--subgraph_hop_n', type=int, default=2, help='k-hop depth for subgraph sampling')
parser.add_argument('--max_subgraph_nodes', type=int, default=64, help='cap subgraph node number; <=0 means no cap')
parser.add_argument('--sample_progress_every', type=int, default=1000, help='sample progress interval')
parser.add_argument('--encoder_filter_enable', type=int, default=1, help='1: run encoder-based clustering/filtering')
parser.add_argument('--encoder_filter_warmup_epochs', type=int, default=200, help='epochs before filtering')
parser.add_argument('--encoder_filter_method', type=str, default='gmm', choices=['gmm', 'kmeans'], help='cluster method')
parser.add_argument('--encoder_filter_clusters', type=int, default=8, help='cluster number')
parser.add_argument('--encoder_filter_top_m_per_cluster', type=int, default=5, help='base keep count in each cluster')
parser.add_argument('--encoder_filter_top_ratio', type=float, default=0.3, help='extra keep ratio over remaining samples')
parser.add_argument('--encoder_filter_score_alpha', type=float, default=1.0, help='weight of novelty score in stage2 filtering')
parser.add_argument('--encoder_filter_score_beta', type=float, default=1.0, help='weight of coverage score in stage2 filtering')
parser.add_argument('--encoder_filter_score_gamma', type=float, default=1.0, help='weight of redundancy score in stage2 filtering')
parser.add_argument('--encoder_filter_force_rebuild', type=int, default=0, help='1: ignore filter cache and rebuild')
parser.add_argument('--encoder_filter_visualize', type=int, default=1, help='1: save encoder-filter clustering visualization')
parser.add_argument('--encoder_filter_visualize_max_points', type=int, default=50000, help='max points for cluster visualization')
parser.add_argument('--encoder_filter_metric_max_points', type=int, default=12000, help='max points for cluster quality metrics')

# Debug/logging
parser.add_argument('--debug_trace_enable', type=int, default=1, help='1: enable stage trace logs')

# Output
parser.add_argument('--result_subdir', type=str, default='', help='subdir under result/')
parser.add_argument('--context_outputs_root', type=str, default='', help='dir for encode/filter artifacts')

args = parser.parse_args()

DEBUG_TRACE_ENABLE = int(args.debug_trace_enable) == 1

if args.max_subgraph_nodes < 0:
    args.max_subgraph_nodes = 0

print('-' * 100)
print(args)
print('-' * 100)
debug_trace(
    'config.pipeline',
    pretrain_method=args.pretrain_method,
    align_method=args.feature_align_method,
    ablation_pre=args.ablation_pre,
    filter_method=args.encoder_filter_method,
)

# Seeds / device
seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(os.getpid())
print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

pretrain_dataset_names = args.pretrain_datasets
shot_num = int(args.shot_num)
drop_percent = float(args.drop_percent)

pretrain_dataset_str = ''
for ds_name in pretrain_dataset_names:
    pretrain_dataset_str += '_' + ds_name

save_dir = os.path.join(parent_directory, 'checkpoints')
run_subdir = args.result_subdir.strip() if args.result_subdir else datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = os.path.join(parent_directory, 'result', run_subdir)
cache_dir = os.path.join(parent_directory, 'cache')
if args.context_outputs_root:
    context_outputs_root = args.context_outputs_root
    if not os.path.isabs(context_outputs_root):
        context_outputs_root = os.path.join(parent_directory, context_outputs_root)
else:
    context_outputs_root = os.path.join(result_dir, 'context_outputs')
context_outputs_root = os.path.abspath(context_outputs_root)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(context_outputs_root, exist_ok=True)

print(f'[result] run_subdir={run_subdir}')
print(f'[result] result_dir={result_dir}')
print(f'[result] context_outputs_root={context_outputs_root}')

logfile = os.path.join(result_dir, 'train.log')
logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)


# ------------------- helper: graph/subgraph -------------------
def _should_log_progress(current, total, every):
    if current == 1 or current == total:
        return True
    if int(every) <= 0:
        return False
    return (int(current) % int(every)) == 0


def _dense_adj_from_sp(adj_sp):
    if sp.isspmatrix(adj_sp):
        return np.asarray(adj_sp.todense())
    return np.asarray(adj_sp)


def _build_nhop_subgraph_nodes(graph_nx, center, hop_n, max_nodes):
    nodes_within = nx.single_source_shortest_path_length(graph_nx, center, cutoff=int(hop_n))
    node_idx = sorted(int(n) for n in nodes_within.keys())
    if int(max_nodes) > 0 and len(node_idx) > int(max_nodes):
        # Keep center and nearest nodes by shortest-path distance for determinism.
        pairs = sorted([(int(v), int(k)) for k, v in nodes_within.items()], key=lambda x: (x[0], x[1]))
        kept = [n for _, n in pairs[: int(max_nodes)]]
        if int(center) not in kept:
            kept[-1] = int(center)
        node_idx = sorted(set(kept))
    return np.array(node_idx, dtype=np.int64)


def _align_feature_graph(feature_np, out_dim, method):
    """Run one graph-level alignment, then slice subgraph features from it.

    Fitting per-subgraph transforms makes feature spaces inconsistent across samples.
    This keeps one shared projection basis for all subgraphs from the same graph.
    """
    in_dim = int(feature_np.shape[1])
    n_nodes = int(feature_np.shape[0])
    eff_dim = int(min(out_dim, in_dim, n_nodes))
    if eff_dim <= 0:
        return torch.zeros((n_nodes, out_dim), dtype=torch.float32)

    method = str(method).lower().strip()
    if method == 'pca':
        compressed_np = pca_compression(feature_np, k=eff_dim)
    else:
        x = np.asarray(feature_np, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        compressed_np = u[:, :eff_dim] * s[:eff_dim]

    compressed = torch.as_tensor(compressed_np, dtype=torch.float32)
    if eff_dim == out_dim:
        return compressed

    pad = torch.zeros((n_nodes, out_dim - eff_dim), dtype=torch.float32)
    return torch.cat([compressed, pad], dim=1)


def _build_subgraph_sample_pool(datasets, target_graph_ids_one_based, unify_dim):
    samples = []
    dataset_sizes = {}

    for dataset_idx, ds_name in enumerate(datasets):
        ds = load_dataset(ds_name)
        graph_ids_zero = [int(g) - 1 for g in target_graph_ids_one_based]
        dataset_sizes[ds_name] = 0

        for gid in graph_ids_zero:
            if gid < 0 or gid >= len(ds):
                print(f'[pool] skip {ds_name}: graph_id={gid + 1} out of range')
                continue

            data_obj = ds[gid]
            full_feature_np, full_adj_sp = process.process_tu(data_obj, data_obj.x.shape[1])
            full_feature_aligned = _align_feature_graph(
                full_feature_np,
                out_dim=unify_dim,
                method=args.feature_align_method,
            )
            debug_trace(
                'align.graph_done',
                dataset=ds_name,
                graph_id=int(gid),
                method=args.feature_align_method,
                in_dim=int(full_feature_np.shape[1]),
                out_dim=int(full_feature_aligned.shape[1]),
                nodes=int(full_feature_aligned.shape[0]),
            )
            full_adj_dense = _dense_adj_from_sp(full_adj_sp)
            g_nx = nx.from_numpy_array(full_adj_dense)

            local_count = 0
            for center in range(data_obj.num_nodes):
                node_idx = _build_nhop_subgraph_nodes(
                    g_nx,
                    center,
                    args.subgraph_hop_n,
                    args.max_subgraph_nodes,
                )
                if node_idx.size < 2:
                    continue

                sub_adj_sp = full_adj_sp[node_idx][:, node_idx]
                sub_feature = full_feature_aligned[node_idx, :]

                sample = {
                    'sample_id': len(samples),
                    'dataset_name': ds_name,
                    'dataset_idx': int(dataset_idx),
                    'graph_id_zero': int(gid),
                    'center_node': int(center),
                    'node_idx': node_idx,
                    'feature': sub_feature,
                    'adj': sub_adj_sp,
                }
                samples.append(sample)
                local_count += 1

                if _should_log_progress(local_count, data_obj.num_nodes, args.sample_progress_every):
                    debug_trace(
                        'pool.sample_progress',
                        dataset=ds_name,
                        graph_id=int(gid),
                        current=local_count,
                        total=int(data_obj.num_nodes),
                        sub_nodes=int(node_idx.size),
                    )

            dataset_sizes[ds_name] += local_count
            debug_trace('pool.graph_done', dataset=ds_name, graph_id=int(gid), samples=local_count)

        debug_trace('pool.dataset_done', dataset=ds_name, samples=dataset_sizes[ds_name])

    if len(samples) == 0:
        raise RuntimeError('No valid subgraph samples were generated.')

    print(f'[pool] total_samples={len(samples)}, per_dataset={dataset_sizes}')
    return samples


def _build_aug_inputs_from_samples(samples, pretrain_method, sparse):
    features = []
    adjs = []
    aug_features = []
    aug_adjs = []
    lbls = []
    bce_masks = []
    sample_dataset_ids = []

    for sample in samples:
        feature = sample['feature']
        adj = sample['adj']

        aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(adj, feature, sparse, drop_percent)

        features.append(feature)
        adjs.append(process.normalize_adj(adj + sp.eye(adj.shape[0])))
        aug_features.append(aug_feature)
        aug_adjs.append(aug_adj)
        lbls.append(lbl)
        bce_masks.append(bce_mask)
        sample_dataset_ids.append(int(sample['dataset_idx']))

    return {
        'features': features,
        'adjs': adjs,
        'aug_features': aug_features,
        'aug_adjs': aug_adjs,
        'lbls': lbls,
        'bce_masks': bce_masks,
        'sample_dataset_ids': sample_dataset_ids,
        'pretrain_method': pretrain_method,
    }


def _to_cuda_pretrain_batch(batch_dict, sparse, pretrain_method):
    out = dict(batch_dict)
    out['adjs_sp'] = list(out['adjs'])

    # For EGRAPHCL, moving the entire 70k+ sample pool to GPU defeats minibatching
    # and can OOM before the first forward pass. Keep data on CPU and let model-side
    # minibatch code stream chunks to GPU.
    if pretrain_method == 'EGRAPHCL' and int(args.egraphcl_keep_data_on_cpu) == 1:
        return out

    out['features'] = [t.cuda() for t in out['features']]
    out['adjs'] = [
        process.sparse_mx_to_torch_sparse_tensor(a).cuda() if sparse else torch.FloatTensor(a.todense()).cuda()
        for a in out['adjs']
    ]

    if pretrain_method in ('GRAPHCL', 'EGRAPHCL'):
        out['aug_features'] = [t.cuda() for t in out['aug_features']]
        out['aug_adjs'] = [t.cuda() for t in out['aug_adjs']]
        out['lbls'] = [t.cuda() for t in out['lbls']]
        out['bce_masks'] = [t.cuda() for t in out['bce_masks']]

    return out


def _encode_sample_with_prompt_encoder(model, sample, sparse):
    dataset_id = int(sample['dataset_idx'])
    seq = sample['feature'].to(device)
    adj_sp = process.normalize_adj(sample['adj'] + sp.eye(sample['adj'].shape[0]))
    if sparse:
        adj_t = process.sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)
    else:
        adj_t = torch.FloatTensor(adj_sp.todense()).to(device)

    str_prompt_layers = model.structure_prompt_layers[dataset_id]
    h = model.gcn(seq, adj_t, sparse, prompt_layers=str_prompt_layers).squeeze(0)

    return h.mean(dim=0)


def _cluster_and_filter_samples_by_encoder(model, samples, sparse, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, 'encoder_filter_cache.pt')

    signature = {
        'filter_visualization_schema': 2,
        'seed': int(args.seed),
        'method': str(args.encoder_filter_method),
        'clusters': int(args.encoder_filter_clusters),
        'top_m_per_cluster': int(args.encoder_filter_top_m_per_cluster),
        'top_ratio': float(args.encoder_filter_top_ratio),
        'score_alpha': float(args.encoder_filter_score_alpha),
        'score_beta': float(args.encoder_filter_score_beta),
        'score_gamma': float(args.encoder_filter_score_gamma),
        'sample_count': int(len(samples)),
        'ablation_pre': str(args.ablation_pre),
        'hop': int(args.subgraph_hop_n),
        'max_subgraph_nodes': int(args.max_subgraph_nodes),
    }

    if int(args.encoder_filter_force_rebuild) != 1 and os.path.exists(cache_path):
        try:
            cached = torch_load_trusted(cache_path)
            if isinstance(cached, dict) and cached.get('signature', None) == signature:
                selected_idx = [int(i) for i in cached.get('selected_idx', [])]
                if len(selected_idx) > 1:
                    before_png = os.path.join(output_dir, 'encoder_filter_clusters_before.png')
                    need_visual_rebuild = int(args.encoder_filter_visualize) == 1 and not os.path.exists(before_png)
                    if not need_visual_rebuild:
                        print(f'[filter] loaded cached selected_idx ({len(selected_idx)}) from {cache_path}')
                        return selected_idx
                    print('[filter] cache matched but visualization is incomplete, rebuilding for plotting')
        except Exception as e:
            print(f'[filter] warn: failed to read cache: {e}')

    model.eval()
    encoded = []
    with torch.no_grad():
        for i, sample in enumerate(samples, start=1):
            emb = _encode_sample_with_prompt_encoder(model, sample, sparse)
            encoded.append(emb.detach().cpu().numpy())
            if _should_log_progress(i, len(samples), args.sample_progress_every):
                debug_trace('filter.encode_progress', current=i, total=len(samples))

    x = np.asarray(encoded, dtype=np.float64)
    k = int(max(2, min(args.encoder_filter_clusters, x.shape[0])))

    if args.encoder_filter_method == 'kmeans':
        cluster_model = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = cluster_model.fit_predict(x)
        centers = np.asarray(cluster_model.cluster_centers_, dtype=np.float64)
    else:
        cluster_model = GaussianMixture(n_components=k, random_state=args.seed)
        labels = cluster_model.fit_predict(x)
        centers = np.asarray(cluster_model.means_, dtype=np.float64)

    cluster_dists = np.linalg.norm(x - centers[labels], axis=1)
    debug_trace('filter.cluster_done', method=args.encoder_filter_method, clusters=int(k), total=int(x.shape[0]))

    stage1_keep = []
    top_m = max(1, int(args.encoder_filter_top_m_per_cluster))
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(cluster_dists[idx])]
        stage1_keep.extend(order[: min(top_m, order.size)].tolist())

    stage1_keep = sorted(set(int(i) for i in stage1_keep))

    all_idx = np.arange(x.shape[0], dtype=np.int64)
    stage1_set = set(stage1_keep)
    remain = np.array([i for i in all_idx.tolist() if int(i) not in stage1_set], dtype=np.int64)

    top_ratio = max(0.0, min(1.0, float(args.encoder_filter_top_ratio)))
    second_take = int(np.ceil(remain.size * top_ratio)) if remain.size > 0 else 0

    if second_take > 0 and remain.size > 0:
        _, _, _, _, _, _, learnability, _ = compute_learnability(
            x=x,
            cluster_ids=labels,
            rep_indices=np.asarray(stage1_keep, dtype=np.int64),
            alpha=float(args.encoder_filter_score_alpha),
            beta=float(args.encoder_filter_score_beta),
            gamma=float(args.encoder_filter_score_gamma),
        )
        order = remain[np.argsort(-learnability[remain])]
        stage2_keep = order[:second_take].tolist()
    else:
        stage2_keep = []

    viz_result = save_encoder_filter_visualization(
        output_dir,
        x,
        labels,
        stage1_keep,
        stage2_keep,
        enabled=(int(args.encoder_filter_visualize) == 1),
    )
    if bool(viz_result.get('saved', False)):
        debug_trace(
            'filter.visualization.saved',
            file=viz_result.get('file', ''),
            before_file=viz_result.get('before_file', ''),
            total_points=int(viz_result.get('total_points', 0)),
            selected_points=int(viz_result.get('selected_points', 0)),
        )
    elif int(args.encoder_filter_visualize) == 1:
        print(f"[filter] warn: failed to save cluster visualization: {viz_result.get('reason', 'unknown')}")

    selected_idx = sorted(set(stage1_keep + [int(i) for i in stage2_keep]))
    if len(selected_idx) < 2:
        # Ensure EGRAPHCL can run.
        fallback = np.argsort(cluster_dists)[: max(2, min(16, x.shape[0]))]
        selected_idx = sorted(set(int(i) for i in fallback.tolist()))

    selected_labels = labels[selected_idx]
    unique_cluster, counts_cluster = np.unique(selected_labels, return_counts=True)
    cluster_count_dict = {str(int(c)): int(n) for c, n in zip(unique_cluster.tolist(), counts_cluster.tolist())}

    quality_all = compute_cluster_quality_metrics(
        x,
        labels,
        max_points=int(args.encoder_filter_metric_max_points),
        seed=int(args.seed),
    )
    quality_selected = compute_cluster_quality_metrics(
        x[selected_idx],
        labels[selected_idx],
        max_points=int(args.encoder_filter_metric_max_points),
        seed=int(args.seed),
    )
    quality_assessment = assess_cluster_quality(quality_all, quality_selected)

    report = {
        'signature': signature,
        'selected_count': int(len(selected_idx)),
        'total_count': int(x.shape[0]),
        'stage1_selected_count': int(len(stage1_keep)),
        'stage2_selected_count': int(len(stage2_keep)),
        'cluster_count_selected': cluster_count_dict,
        'stage2_score_weights': {
            'alpha': float(args.encoder_filter_score_alpha),
            'beta': float(args.encoder_filter_score_beta),
            'gamma': float(args.encoder_filter_score_gamma),
        },
        'cluster_quality_all': quality_all,
        'cluster_quality_selected': quality_selected,
        'cluster_quality_assessment': quality_assessment,
    }

    if bool(quality_all.get('valid', False)):
        print(
            '[filter-metrics] all '
            f"sil={float(quality_all['silhouette']):.4f}, "
            f"dbi={float(quality_all['davies_bouldin']):.4f}, "
            f"chi={float(quality_all['calinski_harabasz']):.2f}, "
            f"pc1/pc2={float(quality_all['pc1_over_pc2_ratio']):.4f}"
        )
    else:
        print(f"[filter-metrics] all invalid: {quality_all.get('reason', 'unknown')}")

    if bool(quality_selected.get('valid', False)):
        print(
            '[filter-metrics] selected '
            f"sil={float(quality_selected['silhouette']):.4f}, "
            f"dbi={float(quality_selected['davies_bouldin']):.4f}, "
            f"chi={float(quality_selected['calinski_harabasz']):.2f}, "
            f"pc1/pc2={float(quality_selected['pc1_over_pc2_ratio']):.4f}"
        )
    else:
        print(f"[filter-metrics] selected invalid: {quality_selected.get('reason', 'unknown')}")

    print(
        '[filter-metrics] assessment '
        f"status={quality_assessment.get('status', 'unknown')}, "
        f"summary={quality_assessment.get('summary', '')}"
    )

    with open(os.path.join(output_dir, 'encoder_filter_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    torch.save(
        {
            'signature': signature,
            'selected_idx': selected_idx,
            'labels': labels,
            'dists': cluster_dists,
        },
        cache_path,
        pickle_protocol=4,
    )

    print(
        f'[filter] method={args.encoder_filter_method}, clusters={k}, '
        f'selected={len(selected_idx)}/{x.shape[0]}, stage1={len(stage1_keep)}, stage2={len(stage2_keep)}'
    )
    print(
        '[filter-stage2] '
        f'top_ratio={float(args.encoder_filter_top_ratio):.4f}, '
        f'weights(alpha,beta,gamma)=('
        f'{float(args.encoder_filter_score_alpha):.4f}, '
        f'{float(args.encoder_filter_score_beta):.4f}, '
        f'{float(args.encoder_filter_score_gamma):.4f})'
    )
    return selected_idx


def _run_pretrain_epochs(model, optimiser, lr_scheduler, pretrain_cuda_batch, start_epoch, end_epoch, save_name):
    best_loss = float('inf')
    best_epoch = -1
    wait = 0
    prev_lr = float(optimiser.param_groups[0]['lr'])

    for epoch in range(start_epoch, end_epoch):
        model.train()
        optimiser.zero_grad()

        total_graphs = int(len(pretrain_cuda_batch['aug_features']))
        chunk_graphs = int(args.egraphcl_epoch_chunk_graphs)
        if chunk_graphs <= 0:
            chunk_graphs = total_graphs
        # _stream_egraphcl_loss accumulates micro-batch losses inside one forward;
        # keeping outer chunk <= micro batch avoids retaining many micro-batch graphs.
        chunk_graphs = min(chunk_graphs, int(args.egraphcl_batch_size))
        chunk_graphs = max(1, min(chunk_graphs, total_graphs))
        num_chunks = int(np.ceil(total_graphs / chunk_graphs))

        loss_value_sum = 0.0
        loss_chunk_count = 0
        non_finite_found = False

        for start in range(0, total_graphs, chunk_graphs):
            end = min(start + chunk_graphs, total_graphs)
            sub_aug_features = pretrain_cuda_batch['aug_features'][start:end]
            sub_aug_adjs = pretrain_cuda_batch['aug_adjs'][start:end]
            sub_lbls = pretrain_cuda_batch['lbls'][start:end]
            sub_bce_masks = pretrain_cuda_batch['bce_masks'][start:end]
            sub_dataset_ids = pretrain_cuda_batch['sample_dataset_ids'][start:end]

            loss_chunk = model(
                sub_aug_features,
                sub_aug_adjs,
                True,
                None,
                None,
                None,
                sub_lbls,
                sub_bce_masks,
                None,
                pretrain_method='EGRAPHCL',
                sample_dataset_ids=sub_dataset_ids,
            )

            if not torch.isfinite(loss_chunk):
                print(f'[train] non-finite chunk loss at epoch {epoch + 1}: {loss_chunk}')
                non_finite_found = True
                break

            # Keep epoch-scale gradient magnitude stable when using chunk accumulation.
            (loss_chunk / float(num_chunks)).backward()
            loss_value_sum += float(loss_chunk.detach().item())
            loss_chunk_count += 1

            del loss_chunk

        if non_finite_found or loss_chunk_count == 0:
            break

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
        optimiser.step()

        loss_value = float(loss_value_sum / max(1, loss_chunk_count))
        if (epoch + 1) <= 5 or (args.epoch_log_every > 0 and ((epoch + 1) % args.epoch_log_every == 0)):
            print(f'Loss:[{loss_value:.8f}]')

        lr_scheduler.step(loss_value)
        current_lr = float(optimiser.param_groups[0]['lr'])
        if current_lr < prev_lr:
            print(f'[train] lr reduced: {prev_lr:.8g} -> {current_lr:.8g} (epoch={epoch + 1})')
            prev_lr = current_lr

        if loss_value < (best_loss - float(args.early_stop_min_delta)):
            best_loss = loss_value
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            wait += 1

        if args.pretrain_method == 'EGRAPHCL':
            stats = getattr(model, 'last_egraphcl_stats', None)
            should_log_mask = (
                (epoch + 1) <= 3
                or (int(args.egraphcl_mask_log_every) > 0 and ((epoch + 1) % int(args.egraphcl_mask_log_every) == 0))
            )
            if should_log_mask and isinstance(stats, dict):
                print(
                    '[egraphcl-mask] '
                    f"branch={stats.get('branch', 'single')} "
                    f"source={stats.get('mask_source', 'encoded_similarity')} "
                    f"batch={int(stats.get('batch_size', args.egraphcl_batch_size))} "
                    f"tau={float(stats.get('tau_used', args.egraphcl_tau)):.4f} "
                    f"mask_ratio={float(stats.get('mask_ratio', 0.0)):.4f} "
                    f"masked={int(stats.get('masked_pairs', 0))}/{int(stats.get('total_pairs', 0))} "
                    f"active_neg={int(stats.get('active_negative_pairs', 0))}"
                )

        if wait >= int(args.patience):
            print(f'Early stopping at epoch={epoch + 1}, patience={args.patience}')
            break

    return {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }


# ------------------- model -------------------
num_pretrain_dataset_num = len(pretrain_dataset_names)
model = PrePrompt(
    args.unify_dim,
    args.hid_units,
    'prelu',
    num_pretrain_dataset_num,
    args.layers_num,
    0.1,
    type_=args.combinetype,
    backbone=args.backbone,
    alpha=args.alpha,
    ablation=args.ablation_pre,
    egraphcl_temperature=args.egraphcl_temperature,
    egraphcl_pair_chunk_size=args.egraphcl_pair_chunk_size,
    egraphcl_candidate_pool=args.egraphcl_candidate_pool,
    egraphcl_soft_topk=args.egraphcl_soft_topk,
    egraphcl_max_negatives=args.egraphcl_max_negatives,
    egraphcl_mask_source=args.egraphcl_mask_source,
    egraphcl_online_tau=args.egraphcl_tau,
    egraphcl_adaptive_tau=args.egraphcl_adaptive_tau,
    egraphcl_target_mask_ratio=args.egraphcl_target_mask_ratio,
    egraphcl_tau_min=args.egraphcl_tau_min,
    egraphcl_tau_max=args.egraphcl_tau_max,
    egraphcl_batch_size=args.egraphcl_batch_size,
    egraphcl_min_negatives_per_anchor=args.egraphcl_min_negatives_per_anchor,
    egraphcl_structure_min_negatives_per_anchor=args.egraphcl_structure_min_negatives_per_anchor,
    egraphcl_max_mask_ratio=args.egraphcl_max_mask_ratio,
).to(device)

set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.feature_align_method}_{args.hid_units}_{args.lr}_{args.backbone}'
)
set_name += (
    f'_encf_{args.encoder_filter_enable}_'
    f'{args.encoder_filter_method}{args.encoder_filter_clusters}_'
    f'm{args.encoder_filter_top_m_per_cluster}_r{args.encoder_filter_top_ratio:.2f}'
)
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')


def _prepare_pretrain_data_all_samples():
    target_graph_ids_one_based = [int(gid) for gid in args.graphId]
    sample_pool = _build_subgraph_sample_pool(pretrain_dataset_names, target_graph_ids_one_based, args.unify_dim)
    batch_dict = _build_aug_inputs_from_samples(sample_pool, args.pretrain_method, sparse=True)
    return sample_pool, batch_dict


if int(args.skip_pretrain) == 1:
    print(f'loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name, map_location=device))
else:
    debug_trace('pretrain.pipeline.start')
    debug_trace('pretrain.prepare.start', method=args.pretrain_method)
    sample_pool_all, pretrain_batch_all = _prepare_pretrain_data_all_samples()
    debug_trace('pretrain.prepare.done', total_samples=len(sample_pool_all))

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',
        factor=float(args.lr_scheduler_factor),
        patience=int(args.lr_scheduler_patience),
        min_lr=float(args.lr_scheduler_min_lr),
    )

    pretrain_cuda_all = _to_cuda_pretrain_batch(pretrain_batch_all, sparse=True, pretrain_method=args.pretrain_method)
    if args.pretrain_method == 'EGRAPHCL':
        print(
            '[egraphcl-data] '
            f'keep_on_cpu={int(args.egraphcl_keep_data_on_cpu) == 1}, '
            f'batch_size={int(args.egraphcl_batch_size)}, total_samples={len(pretrain_batch_all["aug_features"])}'
        )
    debug_trace('pretrain.cuda_batch.ready', samples=len(pretrain_batch_all['features']))

    # Phase-A warmup on all samples.
    warmup_epochs = int(max(0, min(args.nb_epochs, args.encoder_filter_warmup_epochs)))
    if warmup_epochs > 0:
        debug_trace('pretrain.phaseA.start', epochs=warmup_epochs, samples=len(sample_pool_all))
        _run_pretrain_epochs(
            model,
            optimiser,
            lr_scheduler,
            pretrain_cuda_all,
            start_epoch=0,
            end_epoch=warmup_epochs,
            save_name=save_name,
        )
        debug_trace('pretrain.phaseA.done')

    # Phase-B filter by prompt encoder embedding, then continue training.
    use_filter = int(args.encoder_filter_enable) == 1
    if use_filter:
        debug_trace('pretrain.filter.start', method=args.encoder_filter_method)
        filter_dir = os.path.join(context_outputs_root, 'encoder_filter')
        selected_idx = _cluster_and_filter_samples_by_encoder(model, sample_pool_all, sparse=True, output_dir=filter_dir)
        selected_samples = [sample_pool_all[i] for i in selected_idx]
        debug_trace('pretrain.filter.done', selected=len(selected_samples), total=len(sample_pool_all))
    else:
        debug_trace('pretrain.filter.skip', reason='encoder_filter_enable=0 or unsupported method')
        selected_samples = sample_pool_all

    if len(selected_samples) < 2:
        raise RuntimeError('EGRAPHCL requires at least 2 selected samples after filtering.')

    pretrain_batch_selected = _build_aug_inputs_from_samples(selected_samples, args.pretrain_method, sparse=True)
    pretrain_cuda_selected = _to_cuda_pretrain_batch(pretrain_batch_selected, sparse=True, pretrain_method=args.pretrain_method)

    if warmup_epochs < int(args.nb_epochs):
        debug_trace(
            'pretrain.phaseB.start',
            start_epoch=warmup_epochs,
            end_epoch=int(args.nb_epochs),
            samples=len(selected_samples),
        )
        phase_b_stats = _run_pretrain_epochs(
            model,
            optimiser,
            lr_scheduler,
            pretrain_cuda_selected,
            start_epoch=warmup_epochs,
            end_epoch=int(args.nb_epochs),
            save_name=save_name,
        )
        debug_trace(
            'pretrain.phaseB.done',
            best_epoch=phase_b_stats['best_epoch'] + 1 if phase_b_stats['best_epoch'] >= 0 else -1,
            best_loss=phase_b_stats['best_loss'],
        )
    debug_trace('pretrain.pipeline.done', checkpoint=save_name)

print('#' * 50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)
logging.info('#' * 50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)

# ------------------- downstream evaluation -------------------
xent = nn.CrossEntropyLoss()
patience = int(args.patience)
test_idx_num = 100


downstream_dataset = load_dataset(args.dataset)
downstream_loader = DataLoader(downstream_dataset)

for data in downstream_loader:
    print(data)

    features_np, adj_sp = process.process_tu(data, data.x.shape[1])
    print('process done')
    features = _align_feature_graph(features_np, out_dim=args.unify_dim, method=args.feature_align_method).cuda()

    adj_sp = process.normalize_adj(adj_sp + sp.eye(adj_sp.shape[0]))

    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y

    labels_np = np.array(data.y)
    nb_classes = len(np.unique(labels_np))
    print('nb_classes', nb_classes)

    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt_local

        test_subgraph = process.build_subgraph(adj_sp.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt as downprompt_local

    adj = process.sparse_mx_to_torch_sparse_tensor(adj_sp).cuda()

print(f'loading model from {save_name}')
model.load_state_dict(load_state_dict_trusted(save_name))
model = model.cuda()

embeds, _ = model.embed(features, adj, True, None, args.pretrain_method == 'LP')

for downstreamlr in [0.001]:
    test_lbls = labels[idx_test].cuda()

    for shotnum in range(1, shot_num + 1):
        tot = torch.zeros(1).cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)
        print("shotnum", shotnum)

        for i in range(100):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            combines.append(args.beta)

            log = downprompt_local(
                args.hid_units,
                nb_classes,
                args.unify_dim,
                args.layers_num,
                fea_pretext_weights,
                str_pretext_weights,
                combines,
                args.combinetype,
                args.ablation_down,
            ).cuda()
            log.train()

            if args.downstream_task == 'graph':
                idx_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}_graph/{shotnum}-shot_{args.dataset.lower()}/{i}/idx.pt"
                ).type(torch.long).cuda()
                batch_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}_graph/{shotnum}-shot_{args.dataset.lower()}/{i}/batch.pt"
                ).type(torch.long).cuda()
                lbls_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}_graph/{shotnum}-shot_{args.dataset.lower()}/{i}/labels.pt"
                ).type(torch.long).squeeze().cuda()
            else:
                idx_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}/{shotnum}-shot_{args.dataset.lower()}/{i}/idx.pt"
                ).type(torch.long).cuda()
                lbls_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}/{shotnum}-shot_{args.dataset.lower()}/{i}/labels.pt"
                ).type(torch.long).squeeze().cuda()

            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            best = 1e9
            cnt_wait = 0

            for _ in range(400):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj, True, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(features, adj, True, model.gcn, idx_train, lbls_train, 1).float().cuda()

                loss = xent(logits, lbls_train)
                loss_value = float(loss.detach().item())
                if loss_value < best:
                    best = loss_value
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    break

                loss.backward()
                opt.step()

            if args.downstream_task == 'graph':
                logits = log(features, adj, True, model.gcn, test_index, test_batch)
            else:
                logits = log(features, adj, True, model.gcn, idx_test)

            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')

            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)
            tot += acc

        print('-' * 100)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / 100))
        accs_tensor = torch.stack(accs)
        acc_mean = accs_tensor.mean().item()
        acc_std = accs_tensor.std().item()
        microf_mean = sum(microf) / len(microf)
        macrof_mean = sum(macrof) / len(macrof)
        microf_std = torch.std(torch.tensor(microf)).item()
        macrof_std = torch.std(torch.tensor(macrof)).item()
        print('Mean:[{:.4f}]'.format(acc_mean))
        print('Std :[{:.4f}]'.format(acc_std))
        print('-' * 100)

        logging.info('-' * 100)
        logging.info('Mean:[{:.4f}]'.format(acc_mean))
        logging.info('Std :[{:.4f}]'.format(acc_std))
        logging.info('-' * 100)

        with open(csv_name, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, dialect="excel")
            writer.writerow(
                [
                    pretrain_dataset_str,
                    args.dataset,
                    f"{acc_mean:.3f}",
                    f"{acc_std:.3f}",
                    f"{microf_mean:.3f}",
                    f"{microf_std:.3f}",
                    f"{macrof_mean:.3f}",
                    f"{macrof_std:.3f}",
                ]
            )
