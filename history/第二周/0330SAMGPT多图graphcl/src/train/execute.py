from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import random

from models import LogReg
from preprompt import PrePrompt, pca_compression
import preprompt
import pdb
import os
import shutil
import tqdm
import argparse
from datetime import datetime
from downprompt import downprompt, downprompt_graph
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx

# ------------------------------------------------------------
# 本脚本是 SAMGPT 的主入口：
# 1) 从多个预训练数据集上进行预训练（或加载已有 checkpoint）
# 2) 在指定下游数据集上做 few-shot（node 或 graph）分类评测
#
# 主要流程：
# - 解析参数
# - 准备预训练数据（必要时做缓存：feature/adj/aug/...）
# - 训练 PrePrompt 模型（或 skip_pretrain=1 时直接 load）
# - 计算下游数据集 embedding
# - 对每个 shotnum、每个 split(i=0..99) 训练一个 downprompt 分类头并评测
# ------------------------------------------------------------

parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging

# Ensure logs/prints show up promptly when running under nohup (stdout is block-buffered).
# 在某些环境（如 nohup）stdout 默认块缓冲，reconfigure(line_buffering=True) 可提升日志实时性。

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- PyTorch 2.6+ compatibility: torch.load defaults weights_only=True ---
# PyTorch 2.6+ 中 torch.load 默认 weights_only=True。
# 但本项目缓存文件中可能包含 scipy.sparse.csr_matrix 等对象，
# 在 weights_only=True 时会被阻止反序列化导致 UnpicklingError。
# 这里显式对“本地可信文件”使用 weights_only=False。


def torch_load_trusted(path: str, map_location=None):
    """用于加载本地可信缓存/检查点（允许 pickle 任意对象）。"""
    return torch.load(path, map_location=map_location, weights_only=False)


# Convenience helper for state_dict checkpoints.
def load_state_dict_trusted(path: str, map_location=None):
    """加载 state_dict，并做基本类型检查。"""
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd
torch.cuda.empty_cache()
# ------------------- 命令行参数 -------------------
# 基础任务与数据设置
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument(
    '--pretrain_datasets',
    nargs='+',
    type=str,
    help='pretrain datasets',
    default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'],
)
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or EGRAPHCL or LP or splitLP')
# 数据增强与随机性设置
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
# 预训练组合与损失相关设置
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help="target graph's id in one dataset")
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
# 模型结构与优化超参
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
# Stage2 过滤与结构映射设置
parser.add_argument('--max_subgraph_nodes', type=int, default=64, help='unified max subgraph nodes used across all stages')
parser.add_argument('--stage2_top_ratio', type=float, default=0.3, help='top ratio selected by learnability rank in each graph')
parser.add_argument('--context_outputs_root', type=str, default='', help='root dir of unified context outputs')
parser.add_argument('--result_subdir', type=str, default='', help='subdir under result/ for this run; default uses timestamp')
# Stage 结果强制重建开关
parser.add_argument('--force_rebuild_stage1', type=int, default=0, help='force rebuild stage1 encodings')
parser.add_argument('--force_rebuild_stage2', type=int, default=0, help='force rebuild stage2 scores')
# Stage1 自动构建参数
parser.add_argument('--stage1_num_clusters', type=int, default=8, help='number of GMM clusters for stage1 auto-prepare')
parser.add_argument('--stage1_subgraph_hop_n', type=int, default=2, help='n-hop neighborhood depth for stage1 subgraph construction')
parser.add_argument('--stage1_encoding_mode', type=str, default='raw9', choices=['raw9', 'decorrelated7'], help='feature encoding mode used by stage1 clustering')
parser.add_argument('--stage1_use_log1p', type=int, default=1, help='1: log1p skewed count features before scaling in stage1')
parser.add_argument('--stage1_remove_selfloops', type=int, default=1, help='1: remove self-loops in stage1; 0: keep self-loops')
# Stage2 自动构建参数
parser.add_argument('--stage2_top_m_per_cluster', type=int, default=5, help='representative pool size per cluster for stage2 auto-prepare')
parser.add_argument('--stage2_alpha', type=float, default=1.0, help='stage2 novelty weight')
parser.add_argument('--stage2_beta', type=float, default=1.0, help='stage2 coverage weight')
parser.add_argument('--stage2_gamma', type=float, default=1.0, help='stage2 redundancy weight')
# EGRAPHCL 专用参数
parser.add_argument('--egraphcl_tau', type=float, default=0.85, help='soft-positive threshold on structural cosine similarity for EGRAPHCL')
parser.add_argument('--global_subgraph_pool', type=int, default=1, help='1: unified cross-dataset subgraph pool for GRAPHCL/EGRAPHCL')
parser.add_argument('--force_rebuild_global_pool', type=int, default=0, help='1: rebuild merged global subgraph pool cache')
args = parser.parse_args()

args.max_subgraph_nodes = int(args.max_subgraph_nodes)
if args.max_subgraph_nodes <= 0:
    raise ValueError('max_subgraph_nodes must be a positive integer')

if args.pretrain_method == 'EGRAPHCL' and args.ablation_pre != 'all':
    print(
        '[context] EGRAPHCL requires dual-prompt alignment after sample mapping; '
        f'force ablation_pre from {args.ablation_pre} to all.'
    )
    args.ablation_pre = 'all'

# Unified context pipeline always retrains; loading old checkpoint bypasses full flow.
if args.skip_pretrain == 1:
    print('[context] Unified context pipeline forces skip_pretrain=0 to run fresh pretraining.')
    args.skip_pretrain = 0

print('-' * 100)
print(args)
print('-' * 100)
print(
    '[context] flags: '
    'stage2_filter=on, '
    'struct_contrastive=on, '
    f'global_subgraph_pool={args.global_subgraph_pool}, '
    f'skip_pretrain={args.skip_pretrain}'
)

# ------------------- 参数展开/别名 -------------------
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

# 用于命名实验/文件
pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_' + strs

import os

print(os.getpid())
print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 直接指定单卡 id（注意：如果用多进程/多卡 launcher，需要额外处理）
torch.cuda.set_device(args.gpu)

# ------------------- 随机种子 -------------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 路径准备：
# parent_directory := 项目根（SAMGPT 根目录）
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug
from train.stage_pipeline import split_graph_to_subgraphs
from train.stage_pipeline import get_stage1_reference_graph
from train.stage_pipeline import Stage1Config, run_stage1
from train.stage_pipeline import Stage2Config, run_stage2
from train.stage_pipeline import encode_subgraph
from models.stage1_gmm_cluster import fit_clusters
from models.stage2_learnability_model import FEATURE_KEYS, compute_learnability
from layers.stage2_pool_builder import build_representative_pool
from tools.stage_metrics import evaluate_stage1_artifacts, evaluate_stage2_artifacts

# ------------------- 训练/模型超参 -------------------
nb_epochs = 10000
patience = 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
LP = (args.pretrain_method == 'LP')

# 损失函数
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

nonlinearity = 'prelu'  # special name to separate parameters

dataset = args.dataset
device = torch.device("cuda")

# early stopping 相关
best = 1e9
best_t = 0
firstbest = 0
cnt_wait = 0

num_layers_num = args.layers_num

# 预训练数据缓存容器
features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
bce_masks = []
negetive_sample = torch.tensor(0.0)
egraphcl_soft_masked_pairs = 0
egraphcl_negative_pairs = 0
egraphcl_pair_soft_mask = None


def _count_soft_mask_pairs(mask_tensor):
    """Count soft-positive masked pairs from BCE mask's negative half."""
    if mask_tensor is None:
        return 0, 0
    neg_mask = mask_tensor[:, mask_tensor.shape[1] // 2:]
    masked = int((neg_mask <= 0).sum().item())
    total = int(neg_mask.numel())
    return masked, total


print(pretrain_dataset_names)

# 预训练“图”的数量：
# 原实现把 graphId 融入计数（多图数据集/多图选择的情形）
num_pretrain_dataset_num = len(pretrain_dataset_names)

# 为每个预训练数据集创建 DataLoader（本项目多数是单图数据集，但写法比较通用）
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

unify_dim = args.unify_dim

# 日志与输出文件
save_dir = os.path.join(parent_directory, 'checkpoints')
run_subdir = args.result_subdir.strip() if args.result_subdir else datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = os.path.join(parent_directory, 'result', run_subdir)
logfile = os.path.join(result_dir, 'train.log')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
print(f'[result] run_subdir={run_subdir}')
print(f'[result] result_dir={result_dir}')

if args.context_outputs_root:
    if os.path.isabs(args.context_outputs_root):
        context_outputs_root = args.context_outputs_root
    else:
        context_outputs_root = os.path.join(parent_directory, args.context_outputs_root)
else:
    context_outputs_root = os.path.join(result_dir, 'stage_outputs')

context_outputs_root = os.path.abspath(context_outputs_root)
stage1_outputs_root = os.path.join(context_outputs_root, 'stage1')
stage2_outputs_root = os.path.join(context_outputs_root, 'stage2')
os.makedirs(stage1_outputs_root, exist_ok=True)
os.makedirs(stage2_outputs_root, exist_ok=True)


def ensure_context_artifacts(datasets):
    """Ensure stage1/stage2 artifacts exist for unified context pipeline."""
    stage1_root = stage1_outputs_root
    stage2_root = stage2_outputs_root
    for ds in datasets:
        ds_lower = ds.lower()
        ds_stage1_dir = os.path.join(stage1_root, ds_lower)
        stage1_csv = os.path.join(ds_stage1_dir, 'structural_encodings.csv')

        need_build_stage1 = args.force_rebuild_stage1 == 1 or (not os.path.exists(stage1_csv))
        if need_build_stage1:
            print(f'[context] Building stage1 for {ds} -> {ds_stage1_dir}')
            run_stage1(
                Stage1Config(
                    dataset=ds,
                    data_root=os.path.join(parent_directory, 'data'),
                    output_root=stage1_root,
                    max_subgraph_nodes=args.max_subgraph_nodes,
                    subgraph_hop_n=args.stage1_subgraph_hop_n,
                    num_clusters=args.stage1_num_clusters,
                    encoding_mode=args.stage1_encoding_mode,
                    seed=args.seed,
                    remove_selfloops=bool(args.stage1_remove_selfloops),
                    use_log1p=bool(args.stage1_use_log1p),
                )
            )

        ds_stage2_dir = os.path.join(stage2_root, ds_lower)
        stage2_pt = os.path.join(ds_stage2_dir, 'learnability_scores.pt')
        need_build_stage2 = args.force_rebuild_stage2 == 1 or (not os.path.exists(stage2_pt))
        if need_build_stage2:
            if not os.path.exists(stage1_csv):
                raise FileNotFoundError(
                    f'Cannot build stage2 for {ds}: missing stage1 csv at {stage1_csv}'
                )
            print(f'[context] Building stage2 for {ds} -> {ds_stage2_dir}')
            run_stage2(
                Stage2Config(
                    input_csv=stage1_csv,
                    output_dir=ds_stage2_dir,
                    top_m_per_cluster=args.stage2_top_m_per_cluster,
                    alpha=args.stage2_alpha,
                    beta=args.stage2_beta,
                    gamma=args.stage2_gamma,
                )
            )

        # Always refresh stage metrics/visualizations so execute has immediate diagnostics.
        try:
            stage1_eval = evaluate_stage1_artifacts(ds_stage1_dir)
            print(f"[stage1-metrics] {ds}: {stage1_eval['metrics_json']}")
        except Exception as e:
            print(f"[stage1-metrics] warn {ds}: {e}")

        try:
            stage2_eval = evaluate_stage2_artifacts(ds_stage2_dir)
            print(f"[stage2-metrics] {ds}: {stage2_eval['metrics_json']}")
        except Exception as e:
            print(f"[stage2-metrics] warn {ds}: {e}")


if args.global_subgraph_pool != 1:
    ensure_context_artifacts(pretrain_dataset_names)

struct_node_label_cache = {}
struct_cluster_sim_cache = {}


def get_struct_cluster_similarity(dataset_name):
    """Load/build cluster-level structural cosine similarity matrix for EGRAPHCL."""
    key = dataset_name.lower()
    if key in struct_cluster_sim_cache:
        return struct_cluster_sim_cache[key]

    ds_dir = os.path.join(stage1_outputs_root, key)
    stage2_dir = os.path.join(stage2_outputs_root, key)
    os.makedirs(stage2_dir, exist_ok=True)

    sim_npy = os.path.join(stage2_dir, 'egraphcl_cluster_similarity.npy')
    if os.path.exists(sim_npy):
        try:
            sim = np.load(sim_npy)
            struct_cluster_sim_cache[key] = sim
            return sim
        except Exception as e:
            print(f'[egraphcl] warn {dataset_name}: failed to load cached cluster similarity: {e}')

    center_npy = os.path.join(ds_dir, 'cluster_centers.npy')
    if not os.path.exists(center_npy):
        print(f'[egraphcl] skip {dataset_name}: cluster_centers.npy not found at {center_npy}')
        struct_cluster_sim_cache[key] = None
        return None

    try:
        centers = np.asarray(np.load(center_npy), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[0] == 0:
            struct_cluster_sim_cache[key] = None
            return None
        norm = np.linalg.norm(centers, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        c_norm = centers / norm
        sim = c_norm @ c_norm.T
        np.save(sim_npy, sim)
        struct_cluster_sim_cache[key] = sim
        return sim
    except Exception as e:
        print(f'[egraphcl] skip {dataset_name}: failed to build cluster similarity: {e}')
        struct_cluster_sim_cache[key] = None
        return None


def maybe_filter_with_stage2_scores(dataset_name, data_obj, step_id, feature, adj):
    """Optionally filter graph nodes using stage2 learnability pt files.

    Returns:
        feature, adj, is_filtered
    """
    stage2_dir = os.path.join(stage2_outputs_root, dataset_name.lower())
    score_pt_path = os.path.join(stage2_dir, 'learnability_scores.pt')
    pool_pt_path = os.path.join(stage2_dir, 'representative_pool.pt')
    original_nodes = int(feature.size(0))

    if not os.path.exists(score_pt_path):
        print(f'[stage2] skip {dataset_name}: score pt not found at {score_pt_path}')
        return feature, adj, False, None

    try:
        score_pt = torch_load_trusted(score_pt_path)
    except Exception as e:
        print(f'[stage2] skip {dataset_name}: failed to load score pt: {e}')
        return feature, adj, False, None

    graph_ids = score_pt.get('graph_id', None)
    subgraph_ids = score_pt.get('subgraph_id', None)
    rank_desc = score_pt.get('rank_desc', None)
    in_pool = score_pt.get('in_representative_pool', None)
    if graph_ids is None or subgraph_ids is None or rank_desc is None or in_pool is None:
        print(f'[stage2] skip {dataset_name}: score pt missing required keys')
        return feature, adj, False, None

    graph_ids = graph_ids.cpu().numpy().astype(np.int64)
    subgraph_ids = subgraph_ids.cpu().numpy().astype(np.int64)
    rank_desc = rank_desc.cpu().numpy().astype(np.int64)
    in_pool = in_pool.cpu().numpy().astype(np.int64)

    valid_idx = np.where(graph_ids == int(step_id))[0]
    if valid_idx.size == 0:
        print(f'[stage2] skip {dataset_name}: no stage2 samples for graph_id={step_id}')
        return feature, adj, False, None

    pool_selected_sample_idx = []
    rank_selected_sample_idx = []
    selected_sample_idx = []

    def _merge_pool_and_rank_with_quota(pool_idx, rank_idx, rank_base_k):
        # Keep pool first; then scan rank list until unique count reaches pool+rank_base_k.
        pool_unique = []
        seen_pool = set()
        for x in pool_idx:
            xi = int(x)
            if xi not in seen_pool:
                seen_pool.add(xi)
                pool_unique.append(xi)

        selected_unique = list(pool_unique)
        selected_set = set(selected_unique)
        rank_consumed = []
        target_unique = len(pool_unique) + max(0, int(rank_base_k))

        for idx in rank_idx:
            idx = int(idx)
            rank_consumed.append(idx)
            if idx not in selected_set:
                selected_set.add(idx)
                selected_unique.append(idx)

            if len(rank_consumed) >= max(0, int(rank_base_k)) and len(selected_unique) >= target_unique:
                break

        return selected_unique, rank_consumed, target_unique
    if os.path.exists(pool_pt_path):
        try:
            pool_pt = torch_load_trusted(pool_pt_path)
            pool_graph = pool_pt.get('pool_graph_id', None)
            pool_subgraph = pool_pt.get('pool_subgraph_id', None)
            if pool_graph is not None and pool_subgraph is not None:
                pool_graph = pool_graph.cpu().numpy().astype(np.int64)
                pool_subgraph = pool_subgraph.cpu().numpy().astype(np.int64)
                for g_id, sg_id in zip(pool_graph.tolist(), pool_subgraph.tolist()):
                    if g_id == int(step_id):
                        hit = np.where((graph_ids == g_id) & (subgraph_ids == sg_id))[0]
                        for h in hit.tolist():
                            pool_selected_sample_idx.append(int(h))
                            selected_sample_idx.append(int(h))
        except Exception as e:
            print(f'[stage2] warn {dataset_name}: failed to load pool pt: {e}')
    else:
        for i in valid_idx.tolist():
            if in_pool[i] == 1:
                pool_selected_sample_idx.append(int(i))
                selected_sample_idx.append(int(i))

    top_ratio = max(0.0, min(1.0, float(args.stage2_top_ratio)))
    keep_count = max(1, int(np.ceil(valid_idx.size * top_ratio)))
    ranked_valid = [idx for idx in rank_desc.tolist() if graph_ids[idx] == int(step_id)]
    selected_sample_idx, rank_selected_sample_idx, target_unique = _merge_pool_and_rank_with_quota(
        pool_idx=pool_selected_sample_idx,
        rank_idx=ranked_valid,
        rank_base_k=keep_count,
    )

    if len(selected_sample_idx) == 0:
        print(f'[stage2] skip {dataset_name}: no selected samples after filtering')
        return feature, adj, False, None

    # Rebuild stage1-compatible subgraphs for node mapping.
    g = get_stage1_reference_graph(
        dataset_name=dataset_name,
        data_root=os.path.join(parent_directory, 'data'),
        graph_id=int(step_id),
        remove_selfloops=bool(args.stage1_remove_selfloops),
    )
    subgraphs = split_graph_to_subgraphs(
        g,
        max_nodes=args.max_subgraph_nodes,
        hop_n=args.stage1_subgraph_hop_n,
    )

    selected_nodes = set()
    pool_selected_nodes = set()
    rank_selected_nodes = set()
    pool_selected_sample_idx_set = set(pool_selected_sample_idx)
    rank_selected_sample_idx_set = set(rank_selected_sample_idx)
    for idx in selected_sample_idx:
        sg_id = int(subgraph_ids[idx])
        if 0 <= sg_id < len(subgraphs):
            sg_nodes = set(subgraphs[sg_id].nodes())
            selected_nodes.update(sg_nodes)
            if idx in pool_selected_sample_idx_set:
                pool_selected_nodes.update(sg_nodes)
            if idx in rank_selected_sample_idx_set:
                rank_selected_nodes.update(sg_nodes)

    node_overlap_count = len(pool_selected_nodes.intersection(rank_selected_nodes))

    if len(selected_nodes) < 2:
        print(f'[stage2] skip {dataset_name}: selected nodes too few ({len(selected_nodes)})')
        return feature, adj, False, None

    node_idx = np.array(sorted(selected_nodes), dtype=np.int64)
    node_idx_t = torch.tensor(node_idx, dtype=torch.long)
    feature = feature.index_select(0, node_idx_t)
    adj = adj[node_idx][:, node_idx]

    print(
        f'[stage2] {dataset_name} graph_id={step_id}: '
        f'selected_subgraphs={len(selected_sample_idx)}, '
        f'target_unique={target_unique}, '
        f'rank_consumed={len(rank_selected_sample_idx)}, '
        f'pool_nodes={len(pool_selected_nodes)}, '
        f'rank_nodes={len(rank_selected_nodes)}, '
        f'node_overlap={node_overlap_count}, '
        f'selected_nodes={len(node_idx)}, '
        f'original_nodes={original_nodes}, '
        f'keep_ratio={len(node_idx) / max(1, original_nodes):.4f}'
    )
    return feature, adj, True, node_idx


def pca_compression_subgraph(feature_np, out_dim):
    """PCA subgraph features to out_dim, padding zeros when subgraph is small."""
    in_dim = int(feature_np.shape[1])
    n_nodes = int(feature_np.shape[0])
    eff_dim = int(min(out_dim, in_dim, n_nodes))
    if eff_dim <= 0:
        return torch.zeros((n_nodes, out_dim), dtype=torch.float32)

    pca = PCA(n_components=eff_dim)
    compressed = pca.fit_transform(feature_np)
    compressed = torch.FloatTensor(compressed)
    if eff_dim == out_dim:
        return compressed

    pad = torch.zeros((n_nodes, out_dim - eff_dim), dtype=torch.float32)
    return torch.cat([compressed, pad], dim=1)


def build_global_subgraph_pool_records(datasets, target_graph_ids_zero_based):
    """Build merged subgraph records across datasets and run stage2-style selection."""
    cache_path_run = os.path.join(context_outputs_root, 'global_subgraph_pool.pt')
    cache_dir_global = os.path.join(parent_directory, 'cache', 'global_pool')
    os.makedirs(cache_dir_global, exist_ok=True)
    cache_path_global = os.path.join(cache_dir_global, 'global_subgraph_pool.pt')
    run_stage1_dir = os.path.join(stage1_outputs_root, 'global_merged')
    run_stage2_dir = os.path.join(stage2_outputs_root, 'global_merged')
    global_artifacts_root = os.path.join(cache_dir_global, 'artifacts')
    global_stage1_dir_fixed = os.path.join(global_artifacts_root, 'stage1', 'global_merged')
    global_stage2_dir_fixed = os.path.join(global_artifacts_root, 'stage2', 'global_merged')
    global_stage1_csv_fixed = os.path.join(global_stage1_dir_fixed, 'structural_encodings.csv')
    global_stage2_csv_fixed = os.path.join(global_stage2_dir_fixed, 'learnability_scores.csv')
    cache_signature = {
        'datasets': list(datasets),
        'target_graph_ids_zero_based': [int(x) for x in target_graph_ids_zero_based],
        'max_subgraph_nodes': int(args.max_subgraph_nodes),
        'stage1_subgraph_hop_n': int(args.stage1_subgraph_hop_n),
        'stage1_num_clusters': int(args.stage1_num_clusters),
        'stage1_encoding_mode': str(args.stage1_encoding_mode),
        'stage1_use_log1p': int(args.stage1_use_log1p),
        'stage1_remove_selfloops': int(args.stage1_remove_selfloops),
        'stage2_top_m_per_cluster': int(args.stage2_top_m_per_cluster),
        'stage2_alpha': float(args.stage2_alpha),
        'stage2_beta': float(args.stage2_beta),
        'stage2_gamma': float(args.stage2_gamma),
        'stage2_top_ratio': float(args.stage2_top_ratio),
        'stage2_allow_duplicates': 1,
        'seed': int(args.seed),
    }

    if args.force_rebuild_global_pool != 1:
        for cache_path in [cache_path_global, cache_path_run]:
            if not os.path.exists(cache_path):
                continue
            try:
                cached = torch_load_trusted(cache_path)
                if isinstance(cached, dict) and cached.get('signature', None) == cache_signature:
                    selected_samples = cached.get('selected_samples', [])
                    cluster_similarity = np.asarray(cached.get('cluster_similarity', None), dtype=np.float64)
                    if len(selected_samples) > 0 and cluster_similarity is not None:
                        print(
                            f'[global_pool] loaded cache: {cache_path} '
                            f'(selected={len(selected_samples)})'
                        )

                        # Re-materialize diagnostics to current run dir and always refresh metrics/plots.
                        try:
                            if os.path.exists(global_stage1_csv_fixed):
                                os.makedirs(run_stage1_dir, exist_ok=True)
                                shutil.copy2(
                                    global_stage1_csv_fixed,
                                    os.path.join(run_stage1_dir, 'structural_encodings.csv'),
                                )
                            if os.path.exists(global_stage2_csv_fixed):
                                os.makedirs(run_stage2_dir, exist_ok=True)
                                shutil.copy2(
                                    global_stage2_csv_fixed,
                                    os.path.join(run_stage2_dir, 'learnability_scores.csv'),
                                )

                            stage1_eval = evaluate_stage1_artifacts(run_stage1_dir)
                            print(f"[stage1-metrics] global_merged: {stage1_eval['metrics_json']}")
                        except Exception as e:
                            print(f"[stage1-metrics] warn global_merged(cache): {e}")

                        try:
                            stage2_eval = evaluate_stage2_artifacts(run_stage2_dir)
                            print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")
                        except Exception as e:
                            print(f"[stage2-metrics] warn global_merged(cache): {e}")

                        return selected_samples, cluster_similarity
            except Exception as e:
                print(f'[global_pool] warn: failed to load cache {cache_path}: {e}')

    merged_records = []
    merged_meta = []
    sample_id = 0
    dataset_subgraph_counts = {ds: 0 for ds in datasets}

    for dataset_idx, ds in enumerate(datasets):
        for graph_id in target_graph_ids_zero_based:
            try:
                g = get_stage1_reference_graph(
                    dataset_name=ds,
                    data_root=os.path.join(parent_directory, 'data'),
                    graph_id=int(graph_id),
                    remove_selfloops=bool(args.stage1_remove_selfloops),
                )
            except Exception as e:
                print(f'[global_pool] skip {ds} graph_id={graph_id}: {e}')
                continue

            subgraphs = split_graph_to_subgraphs(
                g,
                max_nodes=args.max_subgraph_nodes,
                hop_n=args.stage1_subgraph_hop_n,
            )
            for subgraph_id, subg in enumerate(subgraphs):
                encoded = encode_subgraph(subg)
                rec = {
                    'sample_id': sample_id,
                    'graph_id': int(graph_id),
                    'subgraph_id': int(subgraph_id),
                }
                rec.update(encoded)
                merged_records.append(rec)
                merged_meta.append(
                    {
                        'sample_id': sample_id,
                        'dataset_name': ds,
                        'dataset_idx': int(dataset_idx),
                        'graph_id': int(graph_id),
                        'subgraph_id': int(subgraph_id),
                        'node_idx': np.array(sorted(list(subg.nodes())), dtype=np.int64),
                    }
                )
                sample_id += 1
                dataset_subgraph_counts[ds] += 1

    if len(merged_records) == 0:
        raise RuntimeError('Global subgraph pool is empty. Please check dataset/graphId settings.')

    print(
        f"[debug] subgraph build done: total_subgraphs={len(merged_records)}, "
        f"per_dataset={dataset_subgraph_counts}"
    )

    labels, centers_raw, _, _, _, _ = fit_clusters(
        records=merged_records,
        num_clusters=args.stage1_num_clusters,
        seed=args.seed,
        use_log1p=bool(args.stage1_use_log1p),
        encoding_mode=args.stage1_encoding_mode,
    )
    cluster_ids_tmp, cluster_counts_tmp = np.unique(labels.astype(np.int64), return_counts=True)
    cluster_stat = {
        int(cid): int(cnt)
        for cid, cnt in zip(cluster_ids_tmp.tolist(), cluster_counts_tmp.tolist())
    }
    print(
        f"[debug] clustering done: clusters={len(cluster_ids_tmp)}, "
        f"cluster_counts={cluster_stat}"
    )

    x = np.column_stack([np.array([float(r[k]) for r in merged_records], dtype=np.float64) for k in FEATURE_KEYS])
    cluster_ids = labels.astype(np.int64)
    rep_indices, _ = build_representative_pool(
        x=x,
        cluster_ids=cluster_ids,
        top_m_per_cluster=args.stage2_top_m_per_cluster,
        centers_raw=centers_raw,
    )
    novelty, coverage, redundancy, novelty_n, coverage_n, redundancy_n, learnability, order = compute_learnability(
        x=x,
        cluster_ids=cluster_ids,
        rep_indices=rep_indices,
        alpha=float(args.stage2_alpha),
        beta=float(args.stage2_beta),
        gamma=float(args.stage2_gamma),
    )

    pool_selected_idx = [int(i) for i in rep_indices.tolist()]

    # pool + rank with overlap compensation: use original keep_count from top_ratio.
    pool_unique = []
    seen_pool = set()
    for x in pool_selected_idx:
        xi = int(x)
        if xi not in seen_pool:
            seen_pool.add(xi)
            pool_unique.append(xi)

    selected_idx = list(pool_unique)
    selected_set = set(selected_idx)
    rank_selected_idx = []
    top_ratio = max(0.0, min(1.0, float(args.stage2_top_ratio)))
    keep_count = max(1, int(np.ceil(len(merged_records) * top_ratio)))
    target_unique = len(pool_unique) + keep_count

    for idx in order.tolist():
        idx = int(idx)
        rank_selected_idx.append(idx)
        if idx not in selected_set:
            selected_set.add(idx)
            selected_idx.append(idx)

        if len(rank_selected_idx) >= keep_count and len(selected_idx) >= target_unique:
            break

    print(
        f"[debug] sample pool done: representative_pool={len(rep_indices)}, "
        f"rank_base={keep_count}, "
        f"rank_consumed={len(rank_selected_idx)}, "
        f"target_unique={target_unique}, "
        f"unique_selected={len(selected_idx)}"
    )
    if len(rank_selected_idx) > 0:
        rank_cluster_unique = len(np.unique(cluster_ids[np.asarray(rank_selected_idx, dtype=np.int64)]))
    else:
        rank_cluster_unique = 0
    print(
        f"[debug] rank cluster diversity: rank_selected={len(rank_selected_idx)}, "
        f"unique_clusters={rank_cluster_unique}"
    )

    rep_index_set = set(int(i) for i in rep_indices.tolist())

    # Persist global stage1-like CSV for diagnostics.
    global_stage1_dir = run_stage1_dir
    os.makedirs(global_stage1_dir, exist_ok=True)
    global_stage1_csv = os.path.join(global_stage1_dir, 'structural_encodings.csv')
    stage1_header = [
        'sample_id',
        'graph_id',
        'subgraph_id',
        'num_nodes',
        'num_edges',
        'avg_degree',
        'max_degree',
        'var_degree',
        'density',
        'avg_clustering',
        'triangles',
        'diameter',
        'cluster_id',
    ]
    with open(global_stage1_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(stage1_header) + '\n')
        for i, rec in enumerate(merged_records):
            row = [
                str(int(rec['sample_id'])),
                str(int(rec['graph_id'])),
                str(int(rec['subgraph_id'])),
                str(float(rec['num_nodes'])),
                str(float(rec['num_edges'])),
                str(float(rec['avg_degree'])),
                str(float(rec['max_degree'])),
                str(float(rec['var_degree'])),
                str(float(rec['density'])),
                str(float(rec['avg_clustering'])),
                str(float(rec['triangles'])),
                str(float(rec['diameter'])),
                str(int(cluster_ids[i])),
            ]
            f.write(','.join(row) + '\n')
    selected_samples = []
    for idx in selected_idx:
        meta = merged_meta[idx]
        selected_samples.append(
            {
                'sample_id': int(meta['sample_id']),
                'dataset_name': meta['dataset_name'],
                'dataset_idx': int(meta['dataset_idx']),
                'graph_id': int(meta['graph_id']),
                'subgraph_id': int(meta['subgraph_id']),
                'node_idx': meta['node_idx'],
                'cluster_id': int(cluster_ids[idx]),
                'learnability': float(learnability[idx]),
                'in_representative_pool': int(idx in rep_index_set),
            }
        )

    center_norm = np.linalg.norm(centers_raw, axis=1, keepdims=True)
    center_norm = np.where(center_norm < 1e-12, 1.0, center_norm)
    center_unit = centers_raw / center_norm
    cluster_similarity = center_unit @ center_unit.T

    # Persist global stage2-like CSV for diagnostics.
    global_stage2_dir = run_stage2_dir
    os.makedirs(global_stage2_dir, exist_ok=True)
    global_stage2_csv = os.path.join(global_stage2_dir, 'learnability_scores.csv')
    in_pool = np.zeros(len(merged_records), dtype=np.int64)
    if rep_indices.size > 0:
        in_pool[rep_indices] = 1
    with open(global_stage2_csv, 'w', encoding='utf-8') as f:
        f.write(
            'sample_id,graph_id,subgraph_id,cluster_id,in_representative_pool,novelty,coverage,redundancy,'
            'novelty_norm,coverage_norm,redundancy_norm,learnability\n'
        )
        for idx in order.tolist():
            rec = merged_records[idx]
            row = [
                str(int(rec['sample_id'])),
                str(int(rec['graph_id'])),
                str(int(rec['subgraph_id'])),
                str(int(cluster_ids[idx])),
                str(int(in_pool[idx])),
                str(float(novelty[idx])),
                str(float(coverage[idx])),
                str(float(redundancy[idx])),
                str(float(novelty_n[idx])),
                str(float(coverage_n[idx])),
                str(float(redundancy_n[idx])),
                str(float(learnability[idx])),
            ]
            f.write(','.join(row) + '\n')

    try:
        stage1_eval = evaluate_stage1_artifacts(global_stage1_dir)
        print(f"[stage1-metrics] global_merged: {stage1_eval['metrics_json']}")
    except Exception as e:
        print(f"[stage1-metrics] warn global_merged: {e}")

    try:
        stage2_eval = evaluate_stage2_artifacts(global_stage2_dir)
        print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")
    except Exception as e:
        print(f"[stage2-metrics] warn global_merged: {e}")

    print(
        f'[global_pool] total_subgraphs={len(merged_records)}, '
        f'representative_pool={len(rep_indices)}, '
        f'selected={len(selected_samples)}, '
        f'top_ratio={top_ratio:.4f}'
    )

    cache_payload = {
        'signature': cache_signature,
        'selected_samples': selected_samples,
        'cluster_similarity': cluster_similarity,
    }

    # Persist diagnostics into fixed global folder for future runs.
    try:
        os.makedirs(global_stage1_dir_fixed, exist_ok=True)
        shutil.copy2(global_stage1_csv, global_stage1_csv_fixed)
    except Exception as e:
        print(f'[global_pool] warn: failed to persist stage1 diagnostics to global cache: {e}')

    try:
        os.makedirs(global_stage2_dir_fixed, exist_ok=True)
        shutil.copy2(global_stage2_csv, global_stage2_csv_fixed)
    except Exception as e:
        print(f'[global_pool] warn: failed to persist stage2 diagnostics to global cache: {e}')

    for cache_path in [cache_path_global, cache_path_run]:
        try:
            torch.save(cache_payload, cache_path)
            print(f'[global_pool] saved cache: {cache_path}')
        except Exception as e:
            print(f'[global_pool] warn: failed to save cache {cache_path}: {e}')

    return selected_samples, cluster_similarity


def get_struct_node_labels(dataset_name, data_obj, step_id):
    """Build node->cluster_id mapping from stage1 structural_encodings.csv."""
    cache_key = (dataset_name.lower(), int(step_id))
    if cache_key in struct_node_label_cache:
        return struct_node_label_cache[cache_key]

    csv_path = os.path.join(stage1_outputs_root, dataset_name.lower(), 'structural_encodings.csv')
    if not os.path.exists(csv_path):
        print(f'[struct] skip {dataset_name}: structural_encodings.csv not found at {csv_path}')
        struct_node_label_cache[cache_key] = None
        return None

    try:
        arr = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    except Exception as e:
        print(f'[struct] skip {dataset_name}: failed to read csv: {e}')
        struct_node_label_cache[cache_key] = None
        return None

    if arr.size == 0:
        struct_node_label_cache[cache_key] = None
        return None

    arr = np.atleast_1d(arr)
    graph_ids = arr['graph_id'].astype(np.int64)
    subgraph_ids = arr['subgraph_id'].astype(np.int64)
    cluster_ids = arr['cluster_id'].astype(np.int64)
    use_rows = np.where(graph_ids == int(step_id))[0]
    if use_rows.size == 0:
        struct_node_label_cache[cache_key] = None
        return None

    subgraph_cluster_map = {}
    for ridx in use_rows.tolist():
        subgraph_cluster_map[int(subgraph_ids[ridx])] = int(cluster_ids[ridx])

    g = get_stage1_reference_graph(
        dataset_name=dataset_name,
        data_root=os.path.join(parent_directory, 'data'),
        graph_id=int(step_id),
        remove_selfloops=bool(args.stage1_remove_selfloops),
    )
    subgraphs = split_graph_to_subgraphs(
        g,
        max_nodes=args.max_subgraph_nodes,
        hop_n=args.stage1_subgraph_hop_n,
    )

    node_labels = np.full(data_obj.num_nodes, -1, dtype=np.int64)
    for sg_id, sg in enumerate(subgraphs):
        if sg_id not in subgraph_cluster_map:
            continue
        c_id = subgraph_cluster_map[sg_id]
        node_labels[np.array(list(sg.nodes()), dtype=np.int64)] = c_id

    struct_node_label_cache[cache_key] = node_labels
    return node_labels

# 从 graphId 生成字符串（用于区分多图设置）
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

# 用一堆超参组合出实验名，作为 checkpoint/csv 的文件名
set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
)

# Unified context pipeline naming
set_name += f'_ctx_s2_{args.stage2_top_ratio:.2f}_s3'
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)

# ------------------- 模型构建 -------------------
# PrePrompt: 预训练阶段的主模型（内部通常包含 GNN 编码器 + prompt/融合模块等）
model = PrePrompt(
    unify_dim,
    hid_units,
    nonlinearity,
    num_pretrain_dataset_num,
    num_layers_num,
    0.1,
    type_=args.combinetype,
    backbone=args.backbone,
    alpha=args.alpha,
    ablation=args.ablation_pre,
).cuda()

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
sample_dataset_ids = None
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))
except:
    # ------------------- 预训练数据准备（特征/邻接/增强/负采样）-------------------
    if args.global_subgraph_pool == 1 and args.pretrain_method in ('GRAPHCL', 'EGRAPHCL'):
        target_graph_id_zero_based = [int(gid) - 1 for gid in target_graph_id]
        selected_samples, global_cluster_similarity = build_global_subgraph_pool_records(
            pretrain_dataset_names,
            target_graph_id_zero_based,
        )

        if args.pretrain_method == 'EGRAPHCL':
            selected_cluster_ids = np.asarray([int(s['cluster_id']) for s in selected_samples], dtype=np.int64)
            if (
                global_cluster_similarity is not None
                and selected_cluster_ids.size > 1
                and selected_cluster_ids.max() < global_cluster_similarity.shape[0]
            ):
                pair_sim = global_cluster_similarity[selected_cluster_ids[:, None], selected_cluster_ids[None, :]]
                soft_mask_np = (pair_sim >= float(args.egraphcl_tau)).astype(np.float32)
                np.fill_diagonal(soft_mask_np, 0.0)
                egraphcl_pair_soft_mask = torch.from_numpy(soft_mask_np)

                off_diag_total = int(selected_cluster_ids.size * (selected_cluster_ids.size - 1))
                masked_pairs = int(soft_mask_np.sum())
                compared_pairs = int(off_diag_total - masked_pairs)
                print(
                    '[egraphcl] sample-pair mask summary: '
                    f'samples={selected_cluster_ids.size}, '
                    f'masked_soft_pairs={masked_pairs}, '
                    f'compared_negative_pairs={compared_pairs}, '
                    f'masked_ratio={masked_pairs / max(1, off_diag_total):.6f}'
                )

                per_sample_masked = soft_mask_np.sum(axis=1).astype(np.int64)
                per_sample_total = max(1, int(selected_cluster_ids.size - 1))

                # Print compact cluster-level diagnostics instead of per-sample spam.
                print('[egraphcl] cluster mask stats begin')
                for cid in sorted(np.unique(selected_cluster_ids).tolist()):
                    idxs = np.where(selected_cluster_ids == int(cid))[0]
                    masked_sum = int(per_sample_masked[idxs].sum())
                    total_sum = int(len(idxs) * per_sample_total)
                    compared_sum = int(total_sum - masked_sum)
                    ratio = float(masked_sum / max(1, total_sum))
                    print(
                        '[egraphcl][cluster] '
                        f'cluster_id={int(cid)}, '
                        f'samples={int(len(idxs))}, '
                        f'masked={masked_sum}, '
                        f'compared={compared_sum}, '
                        f'total={total_sum}, '
                        f'masked_ratio={ratio:.6f}'
                    )
                print('[egraphcl] cluster mask stats end')

                # Save visualizations based on sample id and cluster id for offline inspection.
                try:
                    per_sample_ratio = per_sample_masked.astype(np.float64) / float(max(1, per_sample_total))

                    fig = plt.figure(figsize=(11, 4))
                    seq_ids = np.arange(1, int(per_sample_ratio.shape[0]) + 1, dtype=np.int64)
                    plt.plot(seq_ids, per_sample_ratio, linewidth=0.8)
                    plt.xlabel('id (1..N)')
                    plt.ylabel('masked_ratio')
                    plt.title('EGRAPHCL Soft-Mask Ratio by Sequential ID')
                    plt.grid(alpha=0.25)
                    fig.tight_layout()
                    by_id_png = os.path.join(result_dir, 'egraphcl_soft_mask_by_sample_id.png')
                    fig.savefig(by_id_png, dpi=180)
                    plt.close(fig)

                    cid_unique = np.unique(selected_cluster_ids)
                    cid_to_pos = {int(c): i for i, c in enumerate(cid_unique.tolist())}
                    mat = np.zeros((len(cid_unique), len(cid_unique)), dtype=np.float64)
                    cnt = np.zeros((len(cid_unique), len(cid_unique)), dtype=np.float64)
                    for i in range(selected_cluster_ids.size):
                        ci = cid_to_pos[int(selected_cluster_ids[i])]
                        for j in range(selected_cluster_ids.size):
                            if i == j:
                                continue
                            cj = cid_to_pos[int(selected_cluster_ids[j])]
                            mat[ci, cj] += float(soft_mask_np[i, j])
                            cnt[ci, cj] += 1.0
                    mat = np.divide(mat, np.maximum(cnt, 1.0))

                    fig2 = plt.figure(figsize=(6, 5))
                    im = plt.imshow(mat, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='masked_ratio')
                    plt.xticks(range(len(cid_unique)), [str(int(c)) for c in cid_unique], rotation=45)
                    plt.yticks(range(len(cid_unique)), [str(int(c)) for c in cid_unique])
                    plt.xlabel('target cluster_id')
                    plt.ylabel('anchor cluster_id')
                    plt.title('EGRAPHCL Soft-Mask Ratio by Cluster Pair')
                    fig2.tight_layout()
                    by_cluster_png = os.path.join(result_dir, 'egraphcl_soft_mask_by_cluster_id.png')
                    fig2.savefig(by_cluster_png, dpi=180)
                    plt.close(fig2)

                    print(f'[egraphcl] saved visualization: {by_id_png}')
                    print(f'[egraphcl] saved visualization: {by_cluster_png}')
                except Exception as e:
                    print(f'[egraphcl] warn: failed to render soft-mask visualizations: {e}')
            else:
                print('[egraphcl] sample-pair mask summary: unavailable, fallback to default pair mask')

        full_graph_cache = {}
        sample_dataset_ids = []
        for sample in selected_samples:
            ds_name = sample['dataset_name']
            ds_idx = sample['dataset_idx']
            graph_id = sample['graph_id']
            node_idx = sample['node_idx']
            cluster_id = sample['cluster_id']

            cache_key = (ds_name, graph_id)
            if cache_key not in full_graph_cache:
                ds_obj = load_dataset(ds_name)
                if graph_id < 0 or graph_id >= len(ds_obj):
                    raise IndexError(f'graph_id={graph_id} out of range for dataset={ds_name}')
                data_obj = ds_obj[graph_id]
                graph_feature, graph_adj = process.process_tu(data_obj, data_obj.x.shape[1])
                full_graph_cache[cache_key] = (graph_feature, graph_adj)

            graph_feature, graph_adj = full_graph_cache[cache_key]
            sub_feature_raw = graph_feature[node_idx, :]
            sub_adj = graph_adj[node_idx][:, node_idx]

            feature = pca_compression_subgraph(sub_feature_raw, unify_dim)
            adj = process.normalize_adj(sub_adj + sp.eye(sub_adj.shape[0]))

            if args.pretrain_method == 'GRAPHCL':
                aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(adj, feature, sparse, drop_percent)
            else:
                struct_node_labels = np.full(int(feature.shape[0]), int(cluster_id), dtype=np.int64)
                aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(
                    adj,
                    feature,
                    sparse,
                    drop_percent,
                    node_struct_labels=struct_node_labels,
                    cluster_similarity=global_cluster_similarity,
                    soft_positive_tau=args.egraphcl_tau,
                )

            aug_features.append(aug_feature)
            aug_adjs.append(aug_adj)
            lbls.append(lbl)
            bce_masks.append(bce_mask)
            if args.pretrain_method == 'EGRAPHCL' and egraphcl_pair_soft_mask is None:
                masked_pairs, total_pairs = _count_soft_mask_pairs(bce_mask)
                egraphcl_soft_masked_pairs += masked_pairs
                egraphcl_negative_pairs += total_pairs
            features.append(feature)
            adjs.append(adj)
            sample_dataset_ids.append(int(ds_idx))

    else:
        for step, datas in enumerate(zip(*pretrain_loaders)):
            print('step', step)
            # 只训练指定的图 id
            if (step + 1) not in target_graph_id:
                continue

            for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
                # feature/adj 缓存：避免每次运行都重复 process_tu 与 PCA
                if not (
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
                ):
                    feature, adj = process.process_tu(data, data.x.shape[1])
                    # PCA 压缩/统一到 unify_dim 维度
                    feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
                    torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                    torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')

                feature, adj = (
                    torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'),
                    torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_adj.pt'),
                )

                # Optional stage2 filtering on top of cached full-graph tensors.
                feature, adj, stage2_filtered, selected_node_idx = maybe_filter_with_stage2_scores(
                    dataset_name=pretrain_dataset_name,
                    data_obj=data,
                    step_id=step,
                    feature=feature,
                    adj=adj,
                )

                use_egraphcl = (args.pretrain_method == 'EGRAPHCL')
                struct_node_labels = None
                if use_egraphcl:
                    full_struct_labels = get_struct_node_labels(pretrain_dataset_name, data, step)
                    if full_struct_labels is not None:
                        if selected_node_idx is not None:
                            struct_node_labels = full_struct_labels[selected_node_idx]
                        else:
                            struct_node_labels = full_struct_labels

                # ----------- GRAPHCL / EGRAPHCL：需要两份增强视图与对比标签 -----------
                if args.pretrain_method == 'GRAPHCL':
                    if stage2_filtered:
                        # Filtered graph shape changes; do not reuse full-graph augmentation cache.
                        # GRAPHCL keeps vanilla contrastive labels.
                        aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(adj, feature, sparse, drop_percent)
                    elif not (
                        os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_bce_mask.pt')
                    ):
                        aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(adj, feature, sparse, drop_percent)
                        torch.save(aug_feature, f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                        torch.save(aug_adj, f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                        torch.save(lbl, f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                        torch.save(bce_mask, f'{cache_dir}/{pretrain_dataset_name}_bce_mask.pt')
                    else:
                        aug_feature, aug_adj, lbl, bce_mask = (
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_bce_mask.pt'),
                        )
                    aug_features.append(aug_feature)
                    aug_adjs.append(aug_adj)
                    lbls.append(lbl)
                    bce_masks.append(bce_mask)

                elif args.pretrain_method == 'EGRAPHCL':
                    cluster_similarity = get_struct_cluster_similarity(pretrain_dataset_name)

                    if stage2_filtered:
                        # Filtered graph shape changes; do not reuse full-graph augmentation cache.
                        aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(
                            adj,
                            feature,
                            sparse,
                            drop_percent,
                            node_struct_labels=struct_node_labels,
                            cluster_similarity=cluster_similarity,
                            soft_positive_tau=args.egraphcl_tau,
                        )
                    elif not (
                        os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_feature.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_adj.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_lbl.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_bce_mask.pt')
                    ):
                        aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(
                            adj,
                            feature,
                            sparse,
                            drop_percent,
                            node_struct_labels=struct_node_labels,
                            cluster_similarity=cluster_similarity,
                            soft_positive_tau=args.egraphcl_tau,
                        )
                        torch.save(aug_feature, f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_feature.pt')
                        torch.save(aug_adj, f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_adj.pt')
                        torch.save(lbl, f'{cache_dir}/{pretrain_dataset_name}_egraphcl_lbl.pt')
                        torch.save(bce_mask, f'{cache_dir}/{pretrain_dataset_name}_egraphcl_bce_mask.pt')
                    else:
                        aug_feature, aug_adj, lbl, bce_mask = (
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_feature.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_aug_adj.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_lbl.pt'),
                            torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_egraphcl_bce_mask.pt'),
                        )
                    aug_features.append(aug_feature)
                    aug_adjs.append(aug_adj)
                    lbls.append(lbl)
                    bce_masks.append(bce_mask)
                    if egraphcl_pair_soft_mask is None:
                        masked_pairs, total_pairs = _count_soft_mask_pairs(bce_mask)
                        egraphcl_soft_masked_pairs += masked_pairs
                        egraphcl_negative_pairs += total_pairs

                # ----------- splitLP：每个数据集单独做负采样缓存 -----------
                if args.pretrain_method == 'splitLP':
                    if stage2_filtered:
                        negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                    elif not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                        negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                        torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                    else:
                        negetive_sample = torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                    negetive_samples.append(negetive_sample)

                # 邻接统一做归一化并缓存到列表
                adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
                features.append(feature)
                adjs.append(adj)

            # ----------- LP：把多个预训练数据集合并成 block-diagonal 大图后做负采样 -----------
            if args.pretrain_method == 'LP':
                combinedadj = process.combine_dataset_list_sp(adjs)
                print('combinedadj', combinedadj.shape)
                negetive_sample = preprompt.prompt_pretrain_sample(combinedadj, args.negative_samples_num)

    if args.pretrain_method == 'EGRAPHCL':
        if egraphcl_pair_soft_mask is None and egraphcl_negative_pairs > 0:
            kept_pairs = egraphcl_negative_pairs - egraphcl_soft_masked_pairs
            print(
                '[egraphcl] fallback node-level masking summary: '
                f'masked={egraphcl_soft_masked_pairs}, '
                f'kept_negative={kept_pairs}, '
                f'total_negative={egraphcl_negative_pairs}, '
                f'masked_ratio={egraphcl_soft_masked_pairs / egraphcl_negative_pairs:.6f}'
            )
        elif egraphcl_pair_soft_mask is None:
            print('[egraphcl] fallback node-level masking summary: no negative pairs found')
        if egraphcl_pair_soft_mask is not None:
            print(
                '[egraphcl] use sample-level graph-to-graph contrastive loss '
                f'with pair-mask shape={tuple(egraphcl_pair_soft_mask.shape)}'
            )

    # ------------------- 优化器 & 搬到 GPU -------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

        # features: Tensor -> cuda
        features = [tensors.cuda() for tensors in features]
        # adjs: scipy sparse -> torch sparse -> cuda
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
            for adj in adjs
        ]
        lbls = [tensors.cuda() for tensors in lbls]
        bce_masks = [tensors.cuda() for tensors in bce_masks]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        # LP 情况下 negetive_samples 可能为空，此时使用合并图上的 negetive_sample
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ------------------- 预训练循环（early stopping）-------------------
    for epoch in range(nb_epochs):

        loss = 0
        model.train()
        optimiser.zero_grad()

        # GRAPHCL / EGRAPHCL: 传入 (aug_features, aug_adjs, lbls, bce_masks)
        if args.pretrain_method == 'GRAPHCL' or args.pretrain_method == 'EGRAPHCL':
            loss = model(
                aug_features,
                aug_adjs,
                sparse,
                None,
                None,
                None,
                lbls,
                bce_masks,
                None,
                pretrain_method=args.pretrain_method,
                sample_dataset_ids=sample_dataset_ids,
                egraphcl_pair_soft_mask=egraphcl_pair_soft_mask,
            )

        # LP / splitLP: 传入负采样 samples
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss = model(
                features,
                adjs,
                sparse,
                None,
                None,
                None,
                None,
                samples=negetive_samples,
                sample_dataset_ids=sample_dataset_ids,
            )

        loss.backward()
        optimiser.step()

        loss_value = float(loss.detach().item())
        print('Loss:[{:.8f}]'.format(loss_value))
        # 基于 loss 的 early stopping，并保存最优 checkpoint
        if loss_value < best:
            firstbest = 1
            best = loss_value
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            print('Patience:', patience)
            break
        print('Current epoch: {}'.format(epoch + 1))

    print('Best epoch: {}, Best loss: {:.8f}'.format(best_t + 1, best))


# ------------------- 下游评测准备 -------------------
print('#' * 50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)
logging.info('#' * 50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)

# 加载下游数据集并提取特征/邻接
# 注：这里使用 DataLoader 遍历，但大多数情况下 downstream_dataset 只有一张图

downstream_dataset = load_dataset(args.dataset)
print(downstream_dataset)
downstream_loader = DataLoader(downstream_dataset)
for data in downstream_loader:
    print(data)

    features, adj = process.process_tu(data, data.x.shape[1])
    print('process done')
    features = torch.FloatTensor(pca_compression(features, k=unify_dim)).cuda()

    # 归一化 + 自环
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    # 默认取最后 100 个节点作为测试集（不是标准划分，仅作示例/实现设定）
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y

    # 统计类别数
    data = np.array(data.y)
    np.unique(data)
    nb_classes = len(np.unique(data))
    print('nb_classes', nb_classes)

    # downstream_task=graph 时，测试集也要构造子图 batch
    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt

        # 这里传入 dense adjacency（adj.todense().A）并设置 sparse=False
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt

    # 把邻接转 torch sparse（供 GNN 前向）
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

# 加载最优预训练模型并计算 embedding
print(f'loading model from {save_name}')
model.load_state_dict(load_state_dict_trusted(save_name))
model = model.cuda()

# model.embed 返回 embedding；embeds[0, idx] 是节点 idx 的表示
embeds, _ = model.embed(features, adj, sparse, None, LP)

# 下游学习率列表（当前只尝试一个值）
downstreamlrlist = [0.001]

# 测试节点的 embedding（仅用于调试/备用）
test_embs = embeds[0, idx_test]

# ------------------- few-shot 下游评测循环 -------------------
for downstreamlr in downstreamlrlist:
    test_lbls = labels[idx_test].cuda()
    accs = []
    macrof = []
    microf = []
    print('-' * 100)

    for shotnum in range(1, shot_num + 1):
        tot = torch.zeros(1).cuda()
        accs = []
        macrof = []
        microf = []

        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum", shotnum)

        # 100 个 few-shot split（与 generate_fewshot.py 对应）
        for i in tqdm(range(100)):
            # 从预训练模型中取出 prompt/融合相关的权重，让下游头使用
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()

            # 下游 combines 追加 beta（该逻辑依赖 downprompt 的实现约定）
            combines.append(args.beta)

            # 构造下游分类头（downprompt / downprompt_graph）
            log = downprompt(
                hid_units,
                nb_classes,
                unify_dim,
                num_layers_num,
                fea_pretext_weights,
                str_pretext_weights,
                combines,
                args.combinetype,
                args.ablation_down,
            ).cuda()

            log.train()

            # 读取 few-shot 训练集（graph 任务还需读取 batch）
            if args.downstream_task == 'graph':
                idx_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    )
                ).type(torch.long).cuda()

                batch_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    )
                ).type(torch.long).cuda()

                lbls_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    )
                ).type(torch.long).squeeze().cuda()

            else:
                idx_train = torch.load(
                    "data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).cuda()

                lbls_train = torch.load(
                    "data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).squeeze().cuda()

            # 取训练节点 embedding（这里只是取出来，后面真正训练时 downprompt 内部可能还会重新编码）
            pretrain_embs = embeds[0, idx_train]

            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            cnt_wait = 0
            best_acc = torch.zeros(1).cuda()

            # 在 few-shot 训练集上训练分类头，最多 400 step，并用 loss 做 early stopping
            for _ in range(400):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()

                loss = xent(logits, lbls_train)
                loss_value = float(loss.detach().item())
                if loss_value < best:
                    best = loss_value
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    # print('Early stopping!')
                    break

                loss.backward()
                opt.step()

            # 在测试集上推理
            if args.downstream_task == 'graph':
                logits = log(features, adj, sparse, model.gcn, test_index, test_batch)
            else:
                logits = log(features, adj, sparse, model.gcn, idx_test)

            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

            # F1 在 CPU 上用 sklearn 计算
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')

            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)
            tot += acc

        # 统计 100 次 split 的均值/方差并写日志/CSV
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
        logging.info('Mean:[{:.4f}]'.format(accs_tensor.mean().item()))
        logging.info('Std :[{:.4f}]'.format(accs_tensor.std().item()))
        logging.info('-' * 100)

        with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, dialect="excel")

            acc_mean_formatted = f"{acc_mean:.3f}"
            acc_std_formatted = f"{acc_std:.3f}"
            microf_mean_formatted = f"{microf_mean:.3f}"
            macrof_mean_formatted = f"{macrof_mean:.3f}"
            microf_std_formatted = f"{microf_std:.3f}"
            macrof_std_formatted = f"{macrof_std:.3f}"

            writer.writerow(
                [
                    pretrain_dataset_str,
                    downstream_dataset,
                    acc_mean_formatted,
                    acc_std_formatted,
                    microf_mean_formatted,
                    microf_std_formatted,
                    macrof_mean_formatted,
                    macrof_std_formatted,
                ]
            )