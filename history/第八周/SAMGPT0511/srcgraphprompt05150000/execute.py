from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt, pca_compression
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downprompt, downprompt_graph
import csv
import json
import time
from tqdm import tqdm
from usp_sam import ego_subgraph_mask, ego_subgraph_mask_block, _mask_to_csr

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
from torch.cuda.amp import GradScaler, autocast
import logging

# Ensure logs/prints show up promptly when running under nohup (stdout is block-buffered).
# 在某些环境（如 nohup）stdout 默认块缓冲，reconfigure(line_buffering=True) 可提升日志实时性。
try:
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# --- PyTorch 2.6+ compatibility: torch.load defaults weights_only=True ---
# PyTorch 2.6+ 中 torch.load 默认 weights_only=True。
# 但本项目缓存文件中可能包含 scipy.sparse.csr_matrix 等对象，
# 在 weights_only=True 时会被阻止反序列化导致 UnpicklingError。
# 这里显式对“本地可信文件”使用 weights_only=False。


def torch_load_trusted(path: str, map_location=None):
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
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='execution device')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='SAMGPT, USP, GRAPHCL, LP, or splitLP')
parser.add_argument('--source_domains', nargs='+', type=str, default=None, help='alias of pretrain_datasets for minimal USP/SAMGPT runs')
parser.add_argument('--target_domain', type=str, default=None, help='alias of dataset for minimal USP/SAMGPT runs')
parser.add_argument('--shots', nargs='+', type=int, default=None, help='few-shot values for USP; SAMGPT uses max(shots) via shot_num')
parser.add_argument('--subgraph_hop', type=int, default=2, help='USP-SAM k-hop ego-subgraph size')
parser.add_argument(
    '--readout',
    type=str,
    default='prompt_weighted',
    choices=['mean', 'attention', 'prompt_weighted'],
    help='USP-SAM subgraph readout',
)
parser.add_argument('--query_mode', type=str, default='hybrid', choices=['node', 'subgraph', 'hybrid'], help='USP-SAM query type')
parser.add_argument('--prototype', type=str, default='attention', choices=['mean', 'attention'], help='USP-SAM prototype aggregation')
parser.add_argument(
    '--negative_sampling',
    type=str,
    default='domain_balanced',
    choices=['random', 'domain_balanced'],
    help='USP-SAM negative sampling mode; logged for the minimal loop',
)
parser.add_argument('--intra_neg_ratio', type=float, default=0.5, help='USP-SAM intra-domain negative ratio')
parser.add_argument('--num_negatives', type=int, default=128, help='USP-SAM negatives per anchor')
parser.add_argument('--ss_loss_mode', type=str, default='sampled', choices=['full', 'sampled'], help='USP-SAM subgraph-subgraph loss mode')
parser.add_argument('--ss_num_negatives', type=int, default=256, help='USP-SAM sampled negatives per anchor for L_ss')
parser.add_argument('--aug_mask_policy', type=str, default='original_anchor', choices=['original_anchor', 'augmented'], help='USP-SAM readout mask policy for augmented views')
parser.add_argument('--source_batch_size', type=int, default=0, help='number of source graphs per optimizer step; 0 disables batching')
parser.add_argument('--usp_lambda_ns', type=float, default=1.0, help='USP-SAM node-subgraph loss weight')
parser.add_argument('--usp_lambda_ss', type=float, default=1.0, help='USP-SAM subgraph-subgraph loss weight')
parser.add_argument('--usp_lambda_align', type=float, default=None, help='USP-SAM alignment loss weight; default follows SAMGPT alpha')
parser.add_argument('--usp_temperature', type=float, default=0.2, help='USP-SAM InfoNCE temperature')
parser.add_argument('--max_neighbors', type=int, default=64, help='cap ego-subgraph neighbors per node')
parser.add_argument('--compute_local_ns', type=str, default='False', help='compute local node-subgraph InfoNCE in USP forward')
parser.add_argument('--time_profile', type=str, default='True', help='write per-epoch timing profile csv')
parser.add_argument(
    '--progress_detail',
    type=str,
    default='True',
    help='print detailed progress inside each USP epoch',
)
parser.add_argument(
    '--progress_blocks',
    type=str,
    default='False',
    help='show block-level progress for large graph subgraph readout; may be noisy',
)
parser.add_argument(
    '--neg_exclude_scope',
    type=str,
    default='self',
    choices=['ego', 'self', 'none'],
    help='negative sampling exclusion scope',
)
parser.add_argument('--neg_refresh_interval', type=int, default=5, help='epochs between negative index refresh')
parser.add_argument('--use_structure_token', type=str, default='True', help='whether USP-SAM uses SAMGPT structure token')
parser.add_argument('--usp_epochs', type=int, default=20, help='minimal USP-SAM pretraining epochs')
parser.add_argument('--usp_down_epochs', type=int, default=100, help='minimal USP-SAM downstream adaptation epochs')
parser.add_argument(
    '--usp_eval_protocol',
    type=str,
    default='single_split',
    choices=['single_split', 'samgpt_standard'],
    help='USP downstream protocol: one split or SAMGPT-style 100 test nodes over num_splits few-shot splits',
)
parser.add_argument('--epochs', type=int, default=10000, help='SAMGPT pretraining epochs; default preserves original behavior')
parser.add_argument('--num_splits', type=int, default=100, help='few-shot splits to evaluate; default preserves original SAMGPT behavior')
parser.add_argument('--split_start', type=int, default=0, help='first few-shot split id')
parser.add_argument('--align_type', type=str, default='none', choices=['none', 'placeholder_structural_role', 'samgpt_structure_token'])
parser.add_argument('--structure_bucket', type=str, default='mixed', choices=['degree', 'clustering', 'ego_size', 'mixed'])
parser.add_argument('--sanity_calibration', type=str, default='False', help='run small sanity calibration table and exit')
parser.add_argument('--sanity_version', type=str, default='v2', choices=['v2', 'v3'], help='sanity calibration version')
parser.add_argument('--sanity_epochs', nargs='+', type=int, default=[1, 10, 50], help='epochs for sanity calibration')
parser.add_argument('--formal_batch1', type=str, default='False', help='run first minimal formal USP-SAM comparison table and exit')
parser.add_argument('--query_objective_ablation', type=str, default='False', help='run USP query-mode by objective ablation and exit')
parser.add_argument('--formal_seeds', nargs='+', type=int, default=[0, 1, 2], help='seeds for formal batch runs')
parser.add_argument('--formal_epoch', type=int, default=50, help='pretraining epochs for formal batch methods')
parser.add_argument('--overwrite_sanity_table', type=str, default='False', help='overwrite sanity calibration csv before running')
parser.add_argument('--label_shuffle_seeds', nargs='+', type=int, default=[0, 1, 2], help='label-shuffle seeds for sanity checks')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help="target graph's id in one dataset")
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--csr_readout', type=str, default='True', help='use CSR ego list readout to avoid dense masked softmax')
args = parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'1', 'true', 'yes', 'y', 't'}


def resolve_device(device_arg, gpu_id=0):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda was requested, but CUDA is not available.')
        return torch.device(f'cuda:{gpu_id}')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


args.original_pretrain_method = args.pretrain_method
runtime_device = resolve_device(args.device, args.gpu)
if runtime_device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if args.source_domains is not None:
    args.pretrain_datasets = args.source_domains
if args.target_domain is not None:
    args.dataset = args.target_domain
if args.shots is not None:
    args.shot_num = max(args.shots)
else:
    args.shots = [args.shot_num]
args.use_structure_token = str2bool(args.use_structure_token)
args.sanity_calibration = str2bool(args.sanity_calibration)
args.formal_batch1 = str2bool(args.formal_batch1)
args.query_objective_ablation = str2bool(args.query_objective_ablation)
args.overwrite_sanity_table = str2bool(args.overwrite_sanity_table)
args.compute_local_ns = str2bool(args.compute_local_ns)
args.time_profile = str2bool(args.time_profile)
args.progress_detail = str2bool(args.progress_detail)
args.progress_blocks = str2bool(args.progress_blocks)
args.csr_readout = str2bool(args.csr_readout) if hasattr(args, 'csr_readout') else False
if args.neg_refresh_interval < 1:
    args.neg_refresh_interval = 1
if args.usp_lambda_align is None:
    args.usp_lambda_align = args.alpha
if args.pretrain_method == 'SAMGPT':
    args.pretrain_method = 'GRAPHCL'

import warnings

warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

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
if runtime_device.type == 'cuda':
    torch.cuda.set_device(args.gpu)

# ------------------- 随机种子 -------------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn

torch.manual_seed(seed)
if runtime_device.type == 'cuda':
    torch.cuda.manual_seed(seed)

# 路径准备：
# parent_directory := 项目根（src 的上一级）
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug


def _usp_safe_torch_load(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def _usp_device(args):
    return resolve_device(args.device, args.gpu)


def _usp_sparse_to_device(adj, device, sparse=True):
    if sparse:
        return process.sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device)
    return torch.FloatTensor(adj.todense()).to(device)


def _usp_feature_dropout(features, drop_prob):
    if drop_prob <= 0:
        return features
    keep = torch.rand_like(features) > drop_prob
    return features * keep.float()


def _usp_edge_dropout(adj, drop_prob):
    if drop_prob <= 0:
        return adj
    if not adj.is_sparse:
        mask = (torch.rand_like(adj) > drop_prob).float()
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        return adj * mask + eye
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    keep = torch.rand(values.size(0), device=values.device) > drop_prob
    row, col = indices[0], indices[1]
    keep = keep | (row == col)
    new_adj = torch.sparse_coo_tensor(indices[:, keep], values[keep], adj.size(), device=adj.device)
    return new_adj.coalesce()


def _permute_target_edges(adj, seed):
    """Randomly rewire target edges while preserving node count and edge count."""
    if adj.is_sparse:
        dense = adj.to_dense()
    else:
        dense = adj
    device = dense.device
    n = dense.size(0)
    undirected = torch.triu(dense.bool(), diagonal=1)
    edge_count = int(undirected.sum().item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    possible = n * (n - 1) // 2
    edge_count = min(edge_count, possible)
    chosen = torch.randperm(possible, generator=generator)[:edge_count]
    rows = []
    cols = []
    # Invert upper-triangular pair index without materializing all pairs.
    for idx in chosen.tolist():
        remaining = idx
        u = 0
        width = n - 1
        while remaining >= width:
            remaining -= width
            u += 1
            width -= 1
        v = u + 1 + remaining
        rows.append(u)
        cols.append(v)
    rewired = torch.eye(n, dtype=dense.dtype, device=device)
    if rows:
        row = torch.tensor(rows, dtype=torch.long, device=device)
        col = torch.tensor(cols, dtype=torch.long, device=device)
        rewired[row, col] = 1.0
        rewired[col, row] = 1.0
    if adj.is_sparse:
        return rewired.to_sparse().coalesce()
    return rewired


def _identity_like_adj(adj):
    n = adj.size(0)
    device = adj.device
    dtype = adj.dtype
    idx = torch.arange(n, device=device)
    if adj.is_sparse:
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(n, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, adj.size(), device=device).coalesce()
    return torch.eye(n, dtype=dtype, device=device)


def _class_count_json(labels, idx, num_classes):
    selected = labels[idx].detach().cpu()
    counts = torch.bincount(selected.long(), minlength=num_classes).tolist()
    return json.dumps({str(i): int(v) for i, v in enumerate(counts)}, ensure_ascii=False)


def _split_integrity_stats(labels, support_idx, support_labels, query_idx, original_labels=None):
    original_labels = labels if original_labels is None else original_labels
    support_set = set(support_idx.detach().cpu().tolist())
    query_set = set(query_idx.detach().cpu().tolist())
    overlap = len(support_set.intersection(query_set))
    if overlap > 0:
        raise ValueError(f"Support/query split overlap detected: {overlap} nodes.")
    if support_labels is not None and not torch.equal(original_labels[support_idx].long(), support_labels.long()):
        raise ValueError("Few-shot support label file does not match labels[support_idx].")
    if not torch.equal(labels[query_idx].long(), original_labels[query_idx].long()):
        raise ValueError("Label shuffle modified query labels; only support labels may be shuffled.")
    num_classes = int(original_labels.max().item() + 1)
    return {
        'num_support': int(support_idx.numel()),
        'num_query': int(query_idx.numel()),
        'num_classes': num_classes,
        'support_class_count': _class_count_json(labels, support_idx, num_classes),
        'query_class_count': _class_count_json(original_labels, query_idx, num_classes),
        'support_query_overlap': overlap,
        'split_integrity_passed': True,
    }


def _write_rows_csv(path, rows):
    if not rows:
        return
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _write_sanity_summary(path, rows):
    groups = {}
    for row in rows:
        key = (
            row.get('method'),
            row.get('epoch'),
            row.get('label_shuffle'),
            row.get('feature_permuted'),
            row.get('edge_permuted'),
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (method, epoch, label_shuffle, feature_permuted, edge_permuted), group in groups.items():
        acc = np.array([float(r['acc']) for r in group], dtype=float)
        macro = np.array([float(r['macro_f1']) for r in group], dtype=float)
        summary_rows.append({
            'method': method,
            'epoch': epoch,
            'label_shuffle': label_shuffle,
            'feature_permuted': feature_permuted,
            'edge_permuted': edge_permuted,
            'n': len(group),
            'mean_acc': float(acc.mean()),
            'std_acc': float(acc.std(ddof=0)),
            'max_acc': float(acc.max()),
            'p95_acc': float(np.percentile(acc, 95)),
            'mean_macro_f1': float(macro.mean()),
            'std_macro_f1': float(macro.std(ddof=0)),
        })
    _write_rows_csv(path, summary_rows)


def _write_formal_batch1_summary(path, rows):
    groups = {}
    for row in rows:
        key = (row.get('method'), row.get('shot'))
        groups.setdefault(key, []).append(row)
    summary_rows = []
    metrics = ['acc', 'balanced_acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']
    for (method, shot), group in groups.items():
        out = {
            'method': method,
            'shot': shot,
            'n': len(group),
        }
        for metric in metrics:
            values = np.array([float(r[metric]) for r in group], dtype=float)
            out[f'mean_{metric}'] = float(values.mean())
            out[f'std_{metric}'] = float(values.std(ddof=0))
        summary_rows.append(out)
    _write_rows_csv(path, summary_rows)


def _write_query_objective_summary(path, rows):
    groups = {}
    for row in rows:
        key = (
            row.get('method'),
            row.get('objective_active'),
            row.get('query_mode'),
            row.get('shot'),
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    metrics = [
        'acc',
        'balanced_acc',
        'macro_f1',
        'micro_f1',
        'margin',
        'compactness',
        'separation',
        'encoder_param_delta_norm',
        'readout_param_delta_norm',
        'prompt_param_delta_norm',
    ]
    for (method, objective_active, query_mode, shot), group in groups.items():
        out = {
            'method': method,
            'objective_active': objective_active,
            'query_mode': query_mode,
            'shot': shot,
            'n': len(group),
        }
        for metric in metrics:
            values = np.array([float(r[metric]) for r in group], dtype=float)
            out[f'mean_{metric}'] = float(values.mean())
            out[f'std_{metric}'] = float(values.std(ddof=0))
        summary_rows.append(out)
    _write_rows_csv(path, summary_rows)


def _snapshot_named_params(module, name_filter=None):
    snap = {}
    for name, param in module.named_parameters():
        if name_filter is None or name_filter(name, param):
            snap[name] = param.detach().clone()
    return snap


def _delta_norm_from_snapshot(module, snapshot, name_filter=None):
    total = 0.0
    for name, param in module.named_parameters():
        if name not in snapshot:
            continue
        if name_filter is not None and not name_filter(name, param):
            continue
        diff = param.detach() - snapshot[name]
        total += float(diff.pow(2).sum().item())
    return total ** 0.5


def _usp_load_graph(dataset_name, unify_dim, cache_dir, device, sparse=True, data_root=None):
    cache_prefix = os.path.join(cache_dir, dataset_name)
    feature_path = f'{cache_prefix}_feature.pt'
    adj_path = f'{cache_prefix}_adj.pt'

    rebuild_cache = not (os.path.exists(feature_path) and os.path.exists(adj_path))
    if not rebuild_cache:
        cached_feature = _usp_safe_torch_load(feature_path, map_location='cpu')
        rebuild_cache = cached_feature.size(1) != unify_dim

    if rebuild_cache:
        data = load_dataset(dataset_name, path=data_root or './data')[0]
        feature, adj = process.process_tu(data, data.x.shape[1])
        feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
        torch.save(feature, feature_path)
        torch.save(adj, adj_path)

    feature = _usp_safe_torch_load(feature_path).to(device)
    raw_adj = _usp_safe_torch_load(adj_path)
    norm_adj = process.normalize_adj(raw_adj + sp.eye(raw_adj.shape[0]))
    torch_adj = _usp_sparse_to_device(norm_adj, device, sparse=sparse)
    return feature, raw_adj, norm_adj, torch_adj


def _usp_fewshot_path(parent_directory, dataset_name, shot, split):
    base = os.path.join(parent_directory, 'data', f'fewshot_{dataset_name.lower()}', f'{shot}-shot_{dataset_name.lower()}', str(split))
    return os.path.join(base, 'idx.pt'), os.path.join(base, 'labels.pt')


def _usp_read_fewshot(parent_directory, dataset_name, shot, split, device):
    idx_path, label_path = _usp_fewshot_path(parent_directory, dataset_name, shot, split)
    if not (os.path.exists(idx_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f'Missing few-shot split files: {idx_path}, {label_path}')
    idx = _usp_safe_torch_load(idx_path).type(torch.long).to(device)
    labels = _usp_safe_torch_load(label_path).type(torch.long).squeeze().to(device)
    return idx, labels


def _usp_train_downstream_classifier(classifier, node_embeds, subgraph_embeds, support_idx, support_labels, epochs, lr):
    if classifier.prototype == 'mean' and classifier.query != 'hybrid':
        classifier(node_embeds, subgraph_embeds, support_idx, support_idx, support_labels, update_prototypes=True)
        return
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = classifier(
            node_embeds,
            subgraph_embeds,
            support_idx,
            support_idx=support_idx,
            support_labels=support_labels,
            update_prototypes=True,
        )
        loss = F.cross_entropy(logits, support_labels)
        loss.backward()
        opt.step()


def _ego_full_cache_path(cache_dir, name, k, max_neighbors, num_nodes):
    mn = "none" if max_neighbors is None else str(max_neighbors)
    return os.path.join(cache_dir, f"{name}_ego_full_v2_csr_k{k}_max{mn}_N{num_nodes}.pt")


def _ego_block_cache_dir(cache_dir, name, k, max_neighbors, num_nodes, block_size):
    mn = "none" if max_neighbors is None else str(max_neighbors)
    return os.path.join(cache_dir, f"{name}_ego_blocks_v2_csr_k{k}_max{mn}_N{num_nodes}_B{block_size}")


def _cuda_mem_mb(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        return {
            "gpu_allocated_mb": float(torch.cuda.memory_allocated(device) / 1024 / 1024),
            "gpu_reserved_mb": float(torch.cuda.memory_reserved(device) / 1024 / 1024),
        }
    return {
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
    }


def _sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_csv_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _validate_neg_exclude_scope(args, source_graphs, cache_threshold=50_000_000):
    if args.neg_exclude_scope != 'ego':
        return
    print(
        "[NEG-EGO] ego exclusion enabled with CSR ego lists; dense N×N exclude masks are disabled.",
        flush=True,
    )


def _csr_mask_to_row_lists(mask):
    if isinstance(mask, dict):
        indices = mask["ego_indices"]
        indptr = mask["ego_indptr"]
        rows = []
        for r in range(indptr.numel() - 1):
            s = int(indptr[r].item())
            e = int(indptr[r + 1].item())
            rows.append(indices[s:e].clone())
        return rows
    if torch.is_tensor(mask):
        if mask.dim() != 2:
            raise ValueError(f"Expected a 2D mask, got shape {tuple(mask.shape)}")
        rows = []
        for r in range(mask.size(0)):
            rows.append(torch.nonzero(mask[r], as_tuple=False).squeeze(-1).to(torch.long))
        return rows
    raise TypeError(f"Unsupported CSR mask type: {type(mask)}")


def _build_ego_exclusion_rows_from_cache(cache_obj):
    cache_type = cache_obj.get('type')
    if cache_type == 'full':
        return _csr_mask_to_row_lists(cache_obj['mask'])
    if cache_type == 'block':
        rows = []
        for block_path in cache_obj['block_paths']:
            mask_block = _usp_safe_torch_load(block_path, map_location='cpu')
            rows.extend(_csr_mask_to_row_lists(mask_block))
        return rows
    raise ValueError(f"Unsupported ego cache type: {cache_type}")


def _offset_ego_exclusion_rows(rows, offset, device):
    return [row.to(device=device, dtype=torch.long) + offset for row in rows]


def build_or_load_ego_cache(
    name,
    adj,
    cache_dir,
    result_dir,
    k,
    max_neighbors,
    cache_threshold=50_000_000,
    block_size=512,
):
    profile_path = os.path.join(result_dir, "usp_ego_cache_profile.csv")
    t0 = time.time()

    num_nodes = adj.size(0)
    if adj.is_sparse:
        num_edges_or_nnz = int(adj._nnz())
    else:
        num_edges_or_nnz = int(adj.bool().sum().item())

    base_row = {
        "dataset": name,
        "num_nodes": num_nodes,
        "num_edges_or_nnz": num_edges_or_nnz,
        "subgraph_hop": k,
        "max_neighbors": max_neighbors,
        "cache_threshold": cache_threshold,
        "cache_type": None,
        "cache_hit": None,
        "num_blocks": None,
        "avg_ego_size": None,
        "max_ego_size": None,
        "time_sec": None,
        "path": None,
    }

    adj_cpu = adj.to_dense().cpu() if adj.is_sparse else adj.detach().cpu()

    if num_nodes * num_nodes <= cache_threshold:
        mask_path = _ego_full_cache_path(cache_dir, name, k, max_neighbors, num_nodes)
        if os.path.exists(mask_path):
            mask = _usp_safe_torch_load(mask_path, map_location="cpu")
            cache_hit = True
            status = "load_full"
        else:
            dense_mask = ego_subgraph_mask(adj_cpu, k=k, max_neighbors=max_neighbors).cpu()
            mask = _mask_to_csr(dense_mask)
            torch.save(mask, mask_path)
            cache_hit = False
            status = "build_full"

        # compute ego sizes whether mask is CSR dict or dense
        if isinstance(mask, dict):
            indptr = mask["ego_indptr"]
            ego_sizes = (indptr[1:] - indptr[:-1]).float()
        else:
            ego_sizes = mask.sum(dim=1).float()
        row = dict(base_row)
        row.update(
            {
                "cache_type": "full",
                "cache_hit": cache_hit,
                "num_blocks": 1,
                "avg_ego_size": float(ego_sizes.mean().item()),
                "max_ego_size": int(ego_sizes.max().item()),
                "time_sec": time.time() - t0,
                "path": mask_path,
            }
        )
        _append_csv_row(profile_path, row)
        print(
            f"[EGO-CACHE] {name}: {status}, N={num_nodes}, nnz={num_edges_or_nnz}, "
            f"avg_ego={row['avg_ego_size']:.2f}, max_ego={row['max_ego_size']}, "
            f"time={row['time_sec']:.2f}s",
            flush=True,
        )
        return {"type": "full", "mask": mask, "path": mask_path, "cache_format_version": 2}

    block_dir = _ego_block_cache_dir(cache_dir, name, k, max_neighbors, num_nodes, block_size)
    os.makedirs(block_dir, exist_ok=True)
    meta_path = os.path.join(block_dir, "meta.json")
    num_blocks = (num_nodes + block_size - 1) // block_size

    block_paths = []
    ego_sum = 0.0
    ego_max = 0
    cache_hits = 0

    for start in range(0, num_nodes, block_size):
        end = min(start + block_size, num_nodes)
        block_path = os.path.join(block_dir, f"block_{start:08d}_{end:08d}.pt")
        block_paths.append(block_path)
        if os.path.exists(block_path):
            mask_block = _usp_safe_torch_load(block_path, map_location="cpu")
            cache_hits += 1
        else:
            node_idx = torch.arange(start, end, device=adj_cpu.device)
            dense_block = ego_subgraph_mask_block(
                adj_cpu,
                node_idx,
                k=k,
                max_neighbors=max_neighbors,
            ).cpu()
            mask_block = _mask_to_csr(dense_block)
            torch.save(mask_block, block_path)

        # compute ego sizes for this block whether CSR or dense
        if isinstance(mask_block, dict):
            indptr = mask_block["ego_indptr"]
            ego_sizes = (indptr[1:] - indptr[:-1]).float()
        else:
            ego_sizes = mask_block.sum(dim=1).float()
        ego_sum += float(ego_sizes.sum().item())
        ego_max = max(ego_max, int(ego_sizes.max().item()))

    avg_ego = ego_sum / max(1, num_nodes)
    meta = {
        "dataset": name,
        "num_nodes": num_nodes,
        "num_edges_or_nnz": num_edges_or_nnz,
        "k": k,
        "max_neighbors": max_neighbors,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "block_paths": block_paths,
        "avg_ego_size": avg_ego,
        "max_ego_size": ego_max,
        "cache_format_version": 2,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    row = dict(base_row)
    row.update(
        {
            "cache_type": "block",
            "cache_hit": cache_hits == num_blocks,
            "num_blocks": num_blocks,
            "avg_ego_size": avg_ego,
            "max_ego_size": ego_max,
            "time_sec": time.time() - t0,
            "path": block_dir,
        }
    )
    _append_csv_row(profile_path, row)
    print(
        f"[EGO-CACHE] {name}: block_cache, N={num_nodes}, nnz={num_edges_or_nnz}, "
        f"blocks={num_blocks}, hit={cache_hits}/{num_blocks}, avg_ego={avg_ego:.2f}, "
        f"max_ego={ego_max}, time={row['time_sec']:.2f}s",
        flush=True,
    )
    return {
        "type": "block",
        "block_dir": block_dir,
        "block_paths": block_paths,
        "block_size": block_size,
        "num_nodes": num_nodes,
        "meta_path": meta_path,
    }


def run_usp_minimal(args, parent_directory, current_dir, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        ClassPrototypeSubgraphClassifier,
        domain_balanced_negative_indices,
        ego_subgraph_mask_block,
        ego_subgraph_mask,
        prototype_compactness,
        prototype_separation,
        random_negative_indices,
        sampled_info_nce_indexed,
        similarity_margin,
        structural_alignment_loss,
        structural_role_buckets,
    )

    start_time = time.time()

    def _emit_eval_log(message):
        print(message)
        logging.info(message)

    device = _usp_device(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    save_dir = os.path.join(parent_directory, 'checkpoints')
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    source_domains = args.pretrain_datasets
    source_tag = '_'.join(source_domains)
    target_domain = args.dataset
    use_structure_token = args.use_structure_token
    set_name = (
        f'usp_{source_tag}_to_{target_domain}_k{args.subgraph_hop}_'
        f'mn{args.max_neighbors}_{args.readout}_{args.query_mode}_{args.prototype}_seed{args.seed}'
    )
    save_name = os.path.join(save_dir, f'{set_name}.pkl')
    jsonl_name = os.path.join(result_dir, 'usp_minimal_runs.jsonl')
    table_name = os.path.join(result_dir, 'usp_minimal_table.csv')
    time_profile_path = os.path.join(result_dir, 'usp_time_profile.csv')
    time_profile_live_path = os.path.join(result_dir, 'usp_time_profile_live.csv')
    if args.usp_eval_protocol == 'samgpt_standard':
        table_name = os.path.join(result_dir, 'usp_samgpt_standard_eval.csv')

    source_graphs = []
    for name in source_domains:
        feature, _, _, torch_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, feature, torch_adj))

    cached_masks = {}
    cache_threshold = 50_000_000
    for name, _, adj in tqdm(source_graphs, desc='Caching ego-subgraph masks'):
        cached_masks[name] = build_or_load_ego_cache(
            name=name,
            adj=adj,
            cache_dir=cache_dir,
            result_dir=result_dir,
            k=args.subgraph_hop,
            max_neighbors=args.max_neighbors,
            cache_threshold=cache_threshold,
            block_size=512,
        )
    _validate_neg_exclude_scope(args, source_graphs, cache_threshold=cache_threshold)

    model = PrePrompt(
        args.unify_dim,
        args.hid_units,
        'prelu',
        len(source_graphs),
        args.layers_num,
        0.1,
        type_=args.combinetype,
        backbone=args.backbone,
        alpha=args.alpha,
        ablation=args.ablation_pre,
    ).to(device)
    usp_head = USPPretrainingHead(
        args.hid_units,
        readout=args.readout,
        k=args.subgraph_hop,
        temperature=args.usp_temperature,
        lambda_ss=args.usp_lambda_ss,
        lambda_align=args.usp_lambda_align,
        lambda_reg=0.0,
        align_type=args.align_type if args.use_structure_token else 'none',
        structure_bucket=args.structure_bucket,
        max_neighbors=args.max_neighbors,
        use_csr_readout=args.csr_readout,
        ss_loss_mode=args.ss_loss_mode,
        ss_num_negatives=args.ss_num_negatives,
        aug_mask_policy=args.aug_mask_policy,
    ).to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(usp_head.parameters()), lr=args.lr, weight_decay=0.0)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=amp_enabled)
    last_stats = {
        'L_ns_final_epoch_mean': None,
        'L_ss_final_epoch_mean': None,
        'L_align_final_epoch_mean': None,
        'loss_stat_type': None,
        'num_intra_neg': None,
        'num_inter_neg': None,
        'num_negatives_requested': args.num_negatives,
        'num_negatives_effective': None,
        'intra_neg_ratio_requested': args.intra_neg_ratio,
        'intra_neg_ratio_effective': None,
        'negative_fallback_reason': None,
        'negative_sampling_effective': None,
    }
    best_loss = float('inf')
    total_nodes = sum(feature.size(0) for _, feature, _ in source_graphs)
    cached_neg = None
    neg_refresh_count = 0
    neg_reuse_count = 0
    cached_neg_by_batch = {}  # 跨 epoch 共享的缓存，使得 neg_refresh_interval 能正常工作
    for epoch in range(args.usp_epochs):
        epoch_start = time.time()
        forward_time = 0.0
        neg_sampling_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        graph_times = {}
        if args.progress_detail:
            tqdm.write(f"\n[USP] ===== start epoch {epoch + 1}/{args.usp_epochs} =====")
        model.train()
        usp_head.train()
        source_batch_size = args.source_batch_size if args.source_batch_size and args.source_batch_size > 0 else len(source_graphs)
        source_batch_size = max(1, min(source_batch_size, len(source_graphs)))
        epoch_loss_total = 0.0
        epoch_loss_ns_values = []
        epoch_loss_align_values = []
        ss_losses = []
        # neg_debug_enabled = os.environ.get('USP_NEG_DEBUG', '0') == '1'
        neg_debug_enabled = False  # DISABLED: debug output overhead

        # 每个 epoch 对 source_graphs 顺序进行随机打乱，使得跨域负采样更丰富
        # （使用 seed+epoch 保证可复现）
        epoch_order = list(range(len(source_graphs)))
        random.Random(args.seed + epoch).shuffle(epoch_order)

        for batch_start in range(0, len(epoch_order), source_batch_size):
            batch_end = min(batch_start + source_batch_size, len(epoch_order))
            batch_indices = epoch_order[batch_start:batch_end]
            batch_source_graphs = [
                (orig_graph_id, source_graphs[orig_graph_id])
                for orig_graph_id in batch_indices
            ]
            # 使用真实图名作为缓存 key；graph_id 保持为原始 source graph id
            batch_graph_ids = tuple(source_graphs[orig_graph_id][0] for orig_graph_id in batch_indices)
            batch_row_spans = []

            opt.zero_grad(set_to_none=True)
            batch_total_loss = torch.tensor(0.0, device=device)
            batch_h_parts = []
            batch_g_parts = []
            batch_domain_parts = []
            batch_bucket_parts = []
            batch_ego_exclusion_rows = []
            batch_ss_losses = []
            batch_node_offset = 0

            graph_iter = tqdm(
                batch_source_graphs,
                desc=f"USP epoch {epoch + 1}/{args.usp_epochs} graphs[{batch_start}:{batch_end}]",
                leave=False,
                dynamic_ncols=True,
                disable=not args.progress_detail,
            )
            for graph_id, (name, feature, adj) in graph_iter:
                graph_t0 = time.time()
                cache_obj = cached_masks.get(name)
                if isinstance(cache_obj, dict):
                    cache_desc = cache_obj.get("cache_type", cache_obj.get("type", "dict"))
                elif cache_obj is None:
                    cache_desc = "none"
                else:
                    cache_desc = "full_tensor"
                graph_iter.set_postfix_str(f"{name}, N={feature.size(0)}, cache={cache_desc}")
                if args.progress_detail:
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}] graph start: "
                        f"{graph_id + 1}/{len(source_graphs)} {name}, "
                        f"N={feature.size(0)}, cache={cache_desc}"
                    )
                feature_in = feature
                prompt_layers = None
                if use_structure_token:
                    t_prompt = time.time()
                    feature_in = model.feature_prompt_layers[graph_id](feature)
                    prompt_layers = model.structure_prompt_layers[graph_id]
                    if args.progress_detail:
                        tqdm.write(
                            f"[USP][epoch {epoch + 1}][{name}] prompt done, "
                            f"time={time.time() - t_prompt:.2f}s"
                        )

                t_aug = time.time()
                aug_feature_1 = _usp_feature_dropout(feature_in, args.drop_percent)
                aug_feature_2 = _usp_feature_dropout(feature_in, args.drop_percent)
                aug_adj_1 = _usp_edge_dropout(adj, args.drop_percent)
                aug_adj_2 = _usp_edge_dropout(adj, args.drop_percent)
                if args.progress_detail:
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}][{name}] augmentation done, "
                        f"time={time.time() - t_aug:.2f}s"
                    )

                graph_start = time.time()
                out = usp_head(
                    model.gcn,
                    feature_in,
                    adj,
                    sparse=sparse,
                    aug_features_1=aug_feature_1,
                    aug_adj_1=aug_adj_1,
                    aug_features_2=aug_feature_2,
                    aug_adj_2=aug_adj_2,
                    prompt_layers=prompt_layers,
                    align_loss=None,
                    cached_mask=cached_masks.get(name),
                    compute_local_ns=args.compute_local_ns,
                    progress_blocks=args.progress_blocks,
                    progress_prefix=f"[USP][epoch {epoch + 1}][{name}]",
                )
                _sync_if_cuda(device)
                forward_time += time.time() - graph_start
                graph_elapsed = time.time() - graph_t0
                graph_times[name] = time.time() - graph_start
                if args.progress_detail:
                    mem = _cuda_mem_mb(device)
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}][{name}] forward done, "
                        f"time={time.time() - graph_start:.2f}s, "
                        f"L_ss={float(out['loss_ss'].detach()):.6f}, "
                        f"gpu_alloc={mem['gpu_allocated_mb']}"
                    )
                batch_total_loss += args.usp_lambda_ss * out['loss_ss']
                batch_ss_losses.append(float(out['loss_ss'].detach()))
                ss_losses.append(float(out['loss_ss'].detach()))
                batch_h_parts.append(out['node_embeds'])
                batch_g_parts.append(out['subgraph_embeds'])
                batch_domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                batch_row_spans.append((graph_id, name, batch_node_offset, batch_node_offset + out['node_embeds'].size(0)))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    cache_rows = _build_ego_exclusion_rows_from_cache(cache_obj)
                    batch_ego_exclusion_rows.extend(_offset_ego_exclusion_rows(cache_rows, batch_node_offset, device))
                batch_node_offset += out['node_embeds'].size(0)
                if args.align_type == 'placeholder_structural_role':
                    batch_bucket_parts.append(structural_role_buckets(adj, out['subgraph_mask'], bucket=args.structure_bucket))
                if args.progress_detail:
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}] graph done: {name}, "
                        f"total_graph_time={graph_elapsed:.2f}s"
                    )

            concat_start = time.time()
            all_h = torch.cat(batch_h_parts, dim=0)
            all_g = torch.cat(batch_g_parts, dim=0)
            domain_ids = torch.cat(batch_domain_parts, dim=0)
            role_buckets = torch.cat(batch_bucket_parts, dim=0) if batch_bucket_parts else None
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] concat done, "
                    f"all_nodes={all_h.size(0)}, time={time.time() - concat_start:.2f}s"
                )
            neg_start = time.time()
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] negative sampling start: "
                    f"mode={args.negative_sampling}, scope={args.neg_exclude_scope}, "
                    f"refresh_interval={args.neg_refresh_interval}, "
                    f"N={all_h.size(0)}, K={args.num_negatives}"
                )
            if args.negative_sampling == 'domain_balanced':
                if batch_graph_ids not in cached_neg_by_batch or epoch == 0 or (epoch % args.neg_refresh_interval == 0):
                    neg_idx, neg_stats = domain_balanced_negative_indices(
                        domain_ids,
                        num_negatives=args.num_negatives,
                        intra_neg_ratio=args.intra_neg_ratio,
                        ego_exclusion_rows=batch_ego_exclusion_rows if batch_ego_exclusion_rows else None,
                        exclude_scope=args.neg_exclude_scope,
                    )
                    cached_neg_by_batch[batch_graph_ids] = (neg_idx.detach().cpu(), neg_stats)
                    neg_refresh_count += 1
                    neg_event = 'refresh'
                else:
                    cached_neg_idx, neg_stats = cached_neg_by_batch[batch_graph_ids]
                    # 从缓存中复用时，将 CPU 缓存搬回当前设备，避免跨 batch/device 状态问题
                    neg_idx = cached_neg_idx.to(device)
                    neg_reuse_count += 1
                    neg_event = 'reuse'
                negative_sampling_effective = 'domain_balanced'
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
                negative_sampling_effective = 'random'
                neg_event = 'refresh'

            # DISABLED: Pubmed-specific negative sampling debug prints
            # if neg_debug_enabled and any(name == 'Pubmed' for _, name, _, _ in batch_row_spans):
            #     ... [debug code omitted] ...
            _sync_if_cuda(device)
            batch_neg_sampling_time = time.time() - neg_start
            neg_sampling_time += batch_neg_sampling_time
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] negative sampling {neg_event} done, "
                    f"time={batch_neg_sampling_time:.2f}s, "
                    f"num_intra={neg_stats.get('num_intra_neg')}, "
                    f"num_inter={neg_stats.get('num_inter_neg')}, "
                    f"ego_enabled={neg_stats.get('ego_exclusion_enabled')}, "
                    f"avg_ego={neg_stats.get('avg_excluded_ego_size')}, "
                    f"fallback={neg_stats.get('fallback_reason') or neg_stats.get('negative_fallback_reason')}"
                )
            loss_start = time.time()
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                loss_ns, _ = sampled_info_nce_indexed(
                    all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
                )
                align_loss = torch.tensor(0.0, device=device)
                if args.align_type == 'placeholder_structural_role':
                    align_loss = structural_alignment_loss(all_g, domain_ids, role_buckets, temperature=args.usp_temperature)
                elif args.align_type == 'samgpt_structure_token':
                    raise NotImplementedError('samgpt_structure_token alignment is not wired in the minimal loop yet.')
                batch_total_loss = batch_total_loss + args.usp_lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            _sync_if_cuda(device)
            batch_loss_time = time.time() - loss_start
            loss_time += batch_loss_time
            epoch_loss_ns_values.append(float(loss_ns.detach()))
            epoch_loss_align_values.append(float(align_loss.detach()))
            if args.progress_detail:
                mean_ss = float(np.mean(batch_ss_losses)) if batch_ss_losses else 0.0
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] loss done, "
                    f"time={batch_loss_time:.2f}s, "
                    f"L_ns={float(loss_ns.detach()):.6f}, "
                    f"L_ss={mean_ss:.6f}, "
                    f"L_align={float(align_loss.detach()):.6f}"
                )

            if torch.isfinite(batch_total_loss) and batch_total_loss.requires_grad:
                backward_start = time.time()
                scaler.scale(batch_total_loss).backward()
                scaler.step(opt)
                scaler.update()
                _sync_if_cuda(device)
                backward_time += time.time() - backward_start
                if args.progress_detail:
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}] backward/step done, "
                        f"time={time.time() - backward_start:.2f}s"
                    )
            epoch_loss_total += float(batch_total_loss.detach())

        epoch_loss_mean = epoch_loss_total / max(1, len(source_graphs))

        if epoch_loss_mean < best_loss:
            best_loss = epoch_loss_mean
            torch.save({'model': model.state_dict(), 'usp_head': usp_head.state_dict()}, save_name)

        last_stats = {
            'L_ns_final_epoch_mean': float(np.mean(epoch_loss_ns_values)) if epoch_loss_ns_values else None,
            'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
            'L_align_final_epoch_mean': float(np.mean(epoch_loss_align_values)) if epoch_loss_align_values else None,
            'loss_stat_type': 'epoch_mean_over_source_graphs',
            'num_intra_neg': neg_stats.get('num_intra_neg'),
            'num_inter_neg': neg_stats.get('num_inter_neg'),
            'num_negatives_requested': args.num_negatives,
            'num_negatives_effective': neg_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': args.intra_neg_ratio,
            'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
            'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
            'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
            'fallback_reason': neg_stats.get('fallback_reason'),
            'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
            'negative_sampling_effective': negative_sampling_effective,
            'neg_exclude_scope': args.neg_exclude_scope,
            'neg_refresh_interval': args.neg_refresh_interval,
            'neg_refresh_count': neg_refresh_count,
            'neg_reuse_count': neg_reuse_count,
        }
        print(
            f"USP epoch {epoch + 1}/{args.usp_epochs} "
            f"loss_sum={epoch_loss_total:.6f} "
            f"loss={epoch_loss_mean:.6f} "
            f"L_ns={last_stats['L_ns_final_epoch_mean']:.6f} "
            f"L_ss={last_stats['L_ss_final_epoch_mean']:.6f} "
            f"time={time.time() - epoch_start:.2f}s "
            f"neg={neg_sampling_time:.2f}s "
            f"backward={backward_time:.2f}s",
            flush=True,
        )
        if not np.isfinite(epoch_loss_mean):
            raise FloatingPointError('USP-SAM pretraining produced NaN/Inf loss.')

        if args.time_profile:
            mem = _cuda_mem_mb(device)
            _append_csv_row(
                time_profile_live_path,
                {
                    'epoch': epoch + 1,
                    'usp_epochs': args.usp_epochs,
                    'source_domains': '|'.join(source_domains),
                    'target_domain': target_domain,
                    'num_source_graphs': len(source_graphs),
                    'total_source_nodes': total_nodes,
                    'last_batch_nodes': int(all_h.size(0)),
                    'forward_time': forward_time,
                    'negative_sampling_time': neg_sampling_time,
                    'loss_time': loss_time,
                    'backward_time': backward_time,
                    'epoch_time': time.time() - epoch_start,
                    'negative_sampling': args.negative_sampling,
                    'neg_exclude_scope': args.neg_exclude_scope,
                    'neg_refresh_interval': args.neg_refresh_interval,
                    'neg_event': neg_event,
                    'neg_refresh_count': neg_refresh_count,
                    'neg_reuse_count': neg_reuse_count,
                    'num_negatives': args.num_negatives,
                    'intra_neg_ratio': args.intra_neg_ratio,
                    'L_ns': float(loss_ns.detach()),
                    'L_ss': float(np.mean(ss_losses)) if ss_losses else None,
                    'L_align': float(align_loss.detach()),
                    'loss_sum': float(epoch_loss_total),
                    'loss': float(epoch_loss_mean),
                    'graph_times_json': json.dumps(graph_times, ensure_ascii=False),
                    'gpu_allocated_mb': mem['gpu_allocated_mb'],
                    'gpu_reserved_mb': mem['gpu_reserved_mb'],
                },
            )

        if args.time_profile:
            if device.type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            else:
                mem_alloc = None
                mem_reserved = None
            _append_csv_row(
                time_profile_path,
                {
                    'epoch': epoch + 1,
                    'forward_time_sec': forward_time,
                    'neg_sampling_time_sec': neg_sampling_time,
                    'loss_time_sec': loss_time,
                    'backward_time_sec': backward_time,
                    'epoch_total_time_sec': time.time() - epoch_start,
                    'gpu_mem_allocated_mb': mem_alloc,
                    'gpu_mem_reserved_mb': mem_reserved,
                    'num_nodes_total': total_nodes,
                    'neg_refresh_count': neg_refresh_count,
                    'neg_reuse_count': neg_reuse_count,
                },
            )

    if os.path.exists(save_name):
        ckpt = _usp_safe_torch_load(save_name, map_location=device)
        model.load_state_dict(ckpt['model'])
        usp_head.load_state_dict(ckpt['usp_head'])

    feature, raw_adj, _, adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device)
    nb_classes = int(labels.max().item() + 1)
    idx_test = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    test_lbls = labels[idx_test].type(torch.long)

    model.eval()
    usp_head.eval()
    with torch.no_grad():
        node_embeds = usp_head.encode(model.gcn, feature, adj, sparse=sparse, prompt_layers=None)
        subgraph_embeds, _ = usp_head.subgraph_embeddings(node_embeds, adj, k=args.subgraph_hop)

    representation_source = {
        'node': 'node_embedding',
        'subgraph': 'subgraph_readout',
        'hybrid': 'hybrid(node+subgraph)',
    }.get(args.query_mode, f'unknown({args.query_mode})')
    _emit_eval_log(
        f"[EVAL] start downstream_task={args.downstream_task} query_mode={args.query_mode} "
        f"subgraph_hop={args.subgraph_hop} max_neighbors_per_node={args.max_neighbors} "
        f"readout={args.readout} prototype={args.prototype}"
    )
    _emit_eval_log(f"[EVAL] representation_source={representation_source}")

    adj_cpu_dense = adj.to_dense().detach().cpu() if adj.is_sparse else adj.detach().cpu()

    def _subgraph_size_stats(node_idx_tensor):
        if node_idx_tensor is None or node_idx_tensor.numel() == 0:
            return 0.0, 0
        mask_block = ego_subgraph_mask_block(
            adj_cpu_dense,
            node_idx_tensor.detach().cpu(),
            k=args.subgraph_hop,
            max_neighbors=args.max_neighbors,
        )
        sizes = mask_block.sum(dim=1).float()
        return float(sizes.mean().item()), int(sizes.max().item())

    def evaluate_one_split(shot, split):
        support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, split, device)
        _emit_eval_log(
            f"[EVAL] split={split} shot={shot} downstream_task={args.downstream_task} "
            f"query_mode={args.query_mode} subgraph_hop={args.subgraph_hop} "
            f"max_neighbors_per_node={args.max_neighbors}"
        )

        if args.query_mode == 'subgraph':
            avg_support_size, max_support_size = _subgraph_size_stats(support_idx)
            avg_query_size, max_query_size = _subgraph_size_stats(idx_test)
            _emit_eval_log(f"[EVAL] avg_support_subgraph_size={avg_support_size}")
            _emit_eval_log(f"[EVAL] avg_query_subgraph_size={avg_query_size}")
            _emit_eval_log(f"[EVAL] max_subgraph_size_effective={max(max_support_size, max_query_size)}")
            _emit_eval_log(f"[EVAL] support_repr_shape={tuple(subgraph_embeds[support_idx].shape)}")
            _emit_eval_log(f"[EVAL] query_repr_shape={tuple(subgraph_embeds[idx_test].shape)}")
        elif args.query_mode == 'node':
            _emit_eval_log(f"[EVAL] support_repr_shape={tuple(node_embeds[support_idx].shape)}")
            _emit_eval_log(f"[EVAL] query_repr_shape={tuple(node_embeds[idx_test].shape)}")
        else:
            _emit_eval_log(
                f"[EVAL] support_repr_shape=node{tuple(node_embeds[support_idx].shape)}+subgraph{tuple(subgraph_embeds[support_idx].shape)}"
            )
            _emit_eval_log(
                f"[EVAL] query_repr_shape=node{tuple(node_embeds[idx_test].shape)}+subgraph{tuple(subgraph_embeds[idx_test].shape)}"
            )

        classifier = ClassPrototypeSubgraphClassifier(
            args.hid_units,
            nb_classes,
            prototype=args.prototype,
            query=args.query_mode,
            temperature=args.usp_temperature,
            learnable_eta=True,
            eta=0.5,
        ).to(device)
        _usp_train_downstream_classifier(
            classifier,
            node_embeds.detach(),
            subgraph_embeds.detach(),
            support_idx,
            support_labels,
            args.usp_down_epochs,
            args.lr,
        )

        classifier.eval()
        with torch.no_grad():
            classifier.build_prototypes(subgraph_embeds[support_idx], support_labels)
            logits = classifier(node_embeds, subgraph_embeds, idx_test)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == test_lbls).float().mean().item() * 100.0
            preds_cpu = preds.cpu().numpy()
            test_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_cpu, preds_cpu, average='micro') * 100.0
            macro_f1 = f1_score(test_cpu, preds_cpu, average='macro') * 100.0
            margin = similarity_margin(logits, test_lbls).mean().item()
            support_g = subgraph_embeds[support_idx]
            compactness = prototype_compactness(support_g, support_labels, classifier.class_prototypes).item()
            separation = prototype_separation(classifier.class_prototypes).item()
        return {
            'acc': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'margin': margin,
            'compactness': compactness,
            'separation': separation,
        }

    for shot in args.shots:
        if args.usp_eval_protocol == 'samgpt_standard':
            split_ids = range(args.split_start, args.split_start + args.num_splits)
            split_metrics = []
            for split in tqdm(split_ids, desc=f'USP SAMGPT-standard {shot}-shot'):
                split_metrics.append(evaluate_one_split(shot, split))
            metric_keys = ['acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']
            means = {key: float(np.mean([m[key] for m in split_metrics])) for key in metric_keys}
            stds = {f'{key}_std': float(np.std([m[key] for m in split_metrics], ddof=1)) if len(split_metrics) > 1 else 0.0 for key in metric_keys}
            split = f'{args.split_start}-{args.split_start + args.num_splits - 1}'
        else:
            split = args.seed
            means = evaluate_one_split(shot, split)
            stds = {f'{key}_std': None for key in ['acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']}

        row = {
            'method': 'USP',
            'internal_pretrain_path': 'USP',
            'setting': 'source_only',
            'eval_protocol': args.usp_eval_protocol,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': args.seed,
            'split': split,
            'num_splits': args.num_splits if args.usp_eval_protocol == 'samgpt_standard' else 1,
            'test_nodes': int(idx_test.numel()),
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'label_shuffle': False,
            'feature_permuted': False,
            'subgraph_hop': args.subgraph_hop,
            'readout': args.readout,
            'query_mode': args.query_mode,
            'prototype': args.prototype,
            'negative_sampling': args.negative_sampling,
            'neg_exclude_scope': args.neg_exclude_scope,
            'neg_refresh_interval': args.neg_refresh_interval,
            'neg_refresh_count': neg_refresh_count,
            'neg_reuse_count': neg_reuse_count,
            'num_intra_neg': last_stats['num_intra_neg'],
            'num_inter_neg': last_stats['num_inter_neg'],
            'num_negatives_requested': last_stats['num_negatives_requested'],
            'num_negatives_effective': last_stats['num_negatives_effective'],
            'intra_neg_ratio_requested': last_stats['intra_neg_ratio_requested'],
            'intra_neg_ratio_effective': last_stats['intra_neg_ratio_effective'],
            'negative_fallback_reason': last_stats['negative_fallback_reason'],
            'negative_sampling_effective': last_stats['negative_sampling_effective'],
            'use_structure_token': use_structure_token,
            'align_type': args.align_type if args.use_structure_token else 'none',
            'L_ns_final_epoch_mean': last_stats['L_ns_final_epoch_mean'],
            'L_ss_final_epoch_mean': last_stats['L_ss_final_epoch_mean'],
            'L_align_final_epoch_mean': last_stats['L_align_final_epoch_mean'],
            'loss_stat_type': last_stats['loss_stat_type'],
            'train_time': time.time() - start_time,
            'acc': means['acc'],
            'acc_std': stds['acc_std'],
            'macro_f1': means['macro_f1'],
            'macro_f1_std': stds['macro_f1_std'],
            'micro_f1': means['micro_f1'],
            'micro_f1_std': stds['micro_f1_std'],
            'margin': means['margin'],
            'margin_std': stds['margin_std'],
            'compactness': means['compactness'],
            'compactness_std': stds['compactness_std'],
            'separation': means['separation'],
            'separation_std': stds['separation_std'],
        }
        with open(jsonl_name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_header = not os.path.exists(table_name)
        with open(table_name, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print('USP result:', json.dumps(row, ensure_ascii=False))

    return jsonl_name


def _write_sanity_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_sanity_calibration(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        build_subgraph_readout,
        domain_balanced_negative_indices,
        ego_subgraph_mask,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
        structural_alignment_loss,
        structural_role_buckets,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_sanity_calibration_v2.csv')
    if args.overwrite_sanity_table and os.path.exists(table_name):
        os.remove(table_name)
    sanity_rows = []
    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    shot = args.shots[0]
    support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, args.seed, device)
    feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device).long()
    base_query_idx = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    query_idx = base_query_idx
    if torch.isin(support_idx, query_idx).any():
        raise ValueError("Support/query split overlap detected.")
    shuffled_labels_by_seed = {}
    for shuffle_seed in args.label_shuffle_seeds:
        gen = torch.Generator()
        gen.manual_seed(shuffle_seed)
        shuffled_labels = labels.clone()
        perm = torch.randperm(support_labels.numel(), generator=gen).to(device)
        shuffled_labels[support_idx] = support_labels[perm]
        shuffled_labels_by_seed[shuffle_seed] = shuffled_labels
    target_adj_edge_permuted = _permute_target_edges(target_adj, args.seed)

    def emit(
        method,
        internal_path,
        epoch,
        eval_type,
        metrics,
        train_time=0.0,
        loss_stats=None,
        label_shuffle=False,
        feature_permuted=False,
        edge_permuted=False,
        label_shuffle_seed=None,
    ):
        loss_stats = loss_stats or {}
        row = {
            'method': method,
            'internal_pretrain_path': internal_path,
            'epoch': epoch,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': args.seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'eval_type': eval_type,
            'label_shuffle': label_shuffle,
            'label_shuffle_seed': label_shuffle_seed,
            'feature_permuted': feature_permuted,
            'edge_permuted': edge_permuted,
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'acc': metrics['acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        sanity_rows.append(row)
        _write_sanity_row(table_name, row)
        print('SANITY result:', json.dumps(row, ensure_ascii=False))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # FeatureOnly + Prototype: projected input features, no GNN.
    feature_metrics = evaluate_fewshot_prototype(
        feature,
        feature,
        labels,
        support_idx,
        query_idx,
        query_mode='node',
        prototype_type=args.prototype,
        metric='cosine',
    )
    emit('FeatureOnlyPrototype', 'none', 0, 'SharedPrototypeEval', feature_metrics)
    for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
        feature_shuffle_metrics = evaluate_fewshot_prototype(
            feature,
            feature,
            shuffled_labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'FeatureOnlyPrototype_LabelShuffle',
            'none',
            0,
            'SharedPrototypeEval',
            feature_shuffle_metrics,
            label_shuffle=True,
            label_shuffle_seed=shuffle_seed,
        )
    perm = torch.randperm(feature.size(0), device=device)
    feature_permuted = feature[perm]
    feature_perm_metrics = evaluate_fewshot_prototype(
        feature_permuted,
        feature_permuted,
        labels,
        support_idx,
        query_idx,
        query_mode='node',
        prototype_type=args.prototype,
        metric='cosine',
    )
    emit('FeatureOnlyPrototype_FeaturePermuted', 'none', 0, 'SharedPrototypeEval', feature_perm_metrics, feature_permuted=True)
    emit('FeatureOnlyPrototype_EdgePermuted', 'none', 0, 'SharedPrototypeEval', feature_metrics, edge_permuted=True)

    # RandomEncoder + Prototype and NoPretrainGCN + Prototype are intentionally untrained.
    for method in ['RandomEncoderPrototype', 'NoPretrainGCNPrototype']:
        model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            max(1, len(source_domains)),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        model.eval()
        with torch.no_grad():
            h = model.gcn(feature, target_adj, sparse, False).squeeze(0)
            h_edge_permuted = model.gcn(feature, target_adj_edge_permuted, sparse, False).squeeze(0)
        metrics = evaluate_fewshot_prototype(
            h,
            h,
            labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(method, 'none', 0, 'SharedPrototypeEval', metrics)
        if method == 'NoPretrainGCNPrototype':
            for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
                shuffle_metrics = evaluate_fewshot_prototype(
                    h,
                    h,
                    shuffled_labels,
                    support_idx,
                    query_idx,
                    query_mode='node',
                    prototype_type=args.prototype,
                    metric='cosine',
                )
                emit(
                    'NoPretrainGCNPrototype_LabelShuffle',
                    'none',
                    0,
                    'SharedPrototypeEval',
                    shuffle_metrics,
                    label_shuffle=True,
                    label_shuffle_seed=shuffle_seed,
                )
            permuted_h = model.gcn(feature_permuted, target_adj, sparse, False).squeeze(0)
            perm_metrics = evaluate_fewshot_prototype(
                permuted_h,
                permuted_h,
                labels,
                support_idx,
                query_idx,
                query_mode='node',
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit('NoPretrainGCNPrototype_FeaturePermuted', 'none', 0, 'SharedPrototypeEval', perm_metrics, feature_permuted=True)
            edge_metrics = evaluate_fewshot_prototype(
                h_edge_permuted,
                h_edge_permuted,
                labels,
                support_idx,
                query_idx,
                query_mode='node',
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit('NoPretrainGCNPrototype_EdgePermuted', 'none', 0, 'SharedPrototypeEval', edge_metrics, edge_permuted=True)

    source_graphs = []
    for name in source_domains:
        src_feature, _, _, src_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, src_feature, src_adj))

    _validate_neg_exclude_scope(args, source_graphs)

    for epoch_count in args.sanity_epochs:
        # SAMGPT encoder + shared prototype evaluator. This uses GRAPHCL internally.
        start = time.time()
        sam_model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        opt = torch.optim.Adam(sam_model.parameters(), lr=args.lr, weight_decay=0.0)
        graphcl_losses = []
        for _ in range(epoch_count):
            opt.zero_grad()
            total = torch.tensor(0.0, device=device)
            losses = []
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                seq = torch.stack([src_feature, src_feature[torch.randperm(src_feature.size(0))], src_feature.detach(), src_feature.detach()])
                aug1 = _usp_edge_dropout(src_adj, args.drop_percent)
                aug2 = _usp_edge_dropout(src_adj, args.drop_percent)
                adjs = torch.stack([src_adj.to_dense(), aug1.to_dense(), aug2.to_dense()])
                lbl = torch.cat([torch.ones(1, src_feature.size(0)), torch.zeros(1, src_feature.size(0))], dim=1).to(device)
                logits = next(
                    sam_model.compute_prelogits_GRAPHCL(
                        [sam_model.feature_prompt_layers[graph_id]],
                        [sam_model.structure_prompt_layers[graph_id]],
                        [seq],
                        [adjs],
                        sparse=False,
                        msk=None,
                        samp_bias1=None,
                        samp_bias2=None,
                    )
                )
                loss = sam_model.loss(logits, lbl)
                total = total + loss
                losses.append(float(loss.detach()))
            total.backward()
            opt.step()
            graphcl_losses = losses
        sam_model.eval()
        with torch.no_grad():
            sam_h = sam_model.gcn(feature, target_adj, sparse, False).squeeze(0)
        sam_metrics = evaluate_fewshot_prototype(
            sam_h,
            sam_h,
            labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'SAMGPTPrototypeEval',
            'GRAPHCL',
            epoch_count,
            'SharedPrototypeEval',
            sam_metrics,
            train_time=time.time() - start,
            loss_stats={
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': None,
                'L_ss_final_epoch_mean': float(np.mean(graphcl_losses)) if graphcl_losses else None,
                'L_align_final_epoch_mean': None,
            },
        )

        # USP full with shared prototype evaluator.
        start = time.time()
        usp_model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        usp_head = USPPretrainingHead(
            args.hid_units,
            readout=args.readout,
            k=args.subgraph_hop,
            temperature=args.usp_temperature,
            align_type=args.align_type if args.use_structure_token else 'none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        opt = torch.optim.Adam(list(usp_model.parameters()) + list(usp_head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(epoch_count):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts, bucket_parts = [], [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = usp_model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = usp_model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = usp_head(
                    usp_model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + args.usp_lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                bucket_parts.append(structural_role_buckets(src_adj, out['subgraph_mask'], bucket=args.structure_bucket))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    ego_exclusion_rows.extend(_offset_ego_exclusion_rows(
                        _csr_mask_to_row_lists(out['subgraph_mask'].bool()),
                        node_offset,
                        device,
                    ))
                node_offset += out['node_embeds'].size(0)
            all_h = torch.cat(h_parts, dim=0)
            all_g = torch.cat(g_parts, dim=0)
            domain_ids = torch.cat(domain_parts, dim=0)
            role_buckets = torch.cat(bucket_parts, dim=0)
            if args.negative_sampling == 'domain_balanced':
                neg_idx, neg_stats = domain_balanced_negative_indices(
                    domain_ids,
                    num_negatives=args.num_negatives,
                    intra_neg_ratio=args.intra_neg_ratio,
                    ego_exclusion_rows=ego_exclusion_rows if ego_exclusion_rows else None,
                    exclude_scope=args.neg_exclude_scope,
                )
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
            loss_ns, _ = sampled_info_nce_indexed(
                all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
            )
            align_loss = torch.tensor(0.0, device=device)
            if args.align_type == 'placeholder_structural_role':
                align_loss = structural_alignment_loss(all_g, domain_ids, role_buckets, temperature=args.usp_temperature)
            total_loss = total_loss + args.usp_lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            total_loss.backward()
            opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }
        usp_model.eval()
        usp_head.eval()
        with torch.no_grad():
            target_h = usp_head.encode(usp_model.gcn, feature, target_adj, sparse=sparse)
            target_g, _ = usp_head.subgraph_embeddings(target_h, target_adj, k=args.subgraph_hop)
            target_h_edge_permuted = usp_head.encode(usp_model.gcn, feature, target_adj_edge_permuted, sparse=sparse)
            target_g_edge_permuted, _ = usp_head.subgraph_embeddings(
                target_h_edge_permuted,
                target_adj_edge_permuted,
                k=args.subgraph_hop,
            )
        usp_metrics = evaluate_fewshot_prototype(
            target_h,
            target_g,
            labels,
            support_idx,
            query_idx,
            query_mode=args.query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit('USPFull', 'USP', epoch_count, 'SharedPrototypeEval', usp_metrics, time.time() - start, loss_stats)
        for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
            usp_shuffle_metrics = evaluate_fewshot_prototype(
                target_h,
                target_g,
                shuffled_labels,
                support_idx,
                query_idx,
                query_mode=args.query_mode,
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit(
                'USPFull_LabelShuffle',
                'USP',
                epoch_count,
                'SharedPrototypeEval',
                usp_shuffle_metrics,
                time.time() - start,
                loss_stats,
                label_shuffle=True,
                label_shuffle_seed=shuffle_seed,
            )
        usp_edge_metrics = evaluate_fewshot_prototype(
            target_h_edge_permuted,
            target_g_edge_permuted,
            labels,
            support_idx,
            query_idx,
            query_mode=args.query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'USPFull_EdgePermuted',
            'USP',
            epoch_count,
            'SharedPrototypeEval',
            usp_edge_metrics,
            time.time() - start,
            loss_stats,
            edge_permuted=True,
        )

    if sanity_rows:
        tmp_table_name = table_name + '.tmp'
        with open(tmp_table_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sanity_rows)
        os.replace(tmp_table_name, table_name)

    return table_name


def run_sanity_calibration_v3(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        domain_balanced_negative_indices,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
        structural_role_buckets,
        structural_role_features,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_sanity_calibration_v3.csv')
    summary_name = os.path.join(result_dir, 'usp_sanity_calibration_v3_summary.csv')
    if args.overwrite_sanity_table:
        for path in [table_name, summary_name]:
            if os.path.exists(path):
                os.remove(path)

    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    shot = args.shots[0]
    support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, args.seed, device)
    feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    raw_feature = target_data.x.float().to(device)
    labels = target_data.y.to(device).long()
    base_query_idx = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    query_idx = base_query_idx
    base_split_stats = _split_integrity_stats(labels, support_idx, support_labels, query_idx, labels)

    shuffled_labels_by_seed = {}
    for shuffle_seed in args.label_shuffle_seeds:
        gen = torch.Generator()
        gen.manual_seed(shuffle_seed)
        shuffled_labels = labels.clone()
        perm = torch.randperm(support_labels.numel(), generator=gen).to(device)
        shuffled_labels[support_idx] = support_labels[perm]
        _split_integrity_stats(shuffled_labels, support_idx, None, query_idx, labels)
        shuffled_labels_by_seed[shuffle_seed] = shuffled_labels

    target_adj_identity = _identity_like_adj(target_adj)
    target_adj_edge_permuted = _permute_target_edges(target_adj, args.seed)
    rows = []

    def emit(
        method,
        internal_path,
        epoch,
        eval_type,
        metrics,
        train_time=0.0,
        loss_stats=None,
        label_shuffle=False,
        label_shuffle_seed=None,
        feature_permuted=False,
        edge_permuted=False,
        target_edges_used=True,
        split_labels=None,
    ):
        loss_stats = loss_stats or {}
        split_labels = labels if split_labels is None else split_labels
        split_stats = _split_integrity_stats(split_labels, support_idx, None, query_idx, labels)
        row = {
            'method': method,
            'internal_pretrain_path': internal_path,
            'epoch': epoch,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': args.seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': target_edges_used,
            'eval_type': eval_type,
            'label_shuffle': label_shuffle,
            'label_shuffle_seed': label_shuffle_seed,
            'feature_permuted': feature_permuted,
            'edge_permuted': edge_permuted,
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'negative_sampling': args.negative_sampling,
            'neg_exclude_scope': args.neg_exclude_scope,
            'neg_refresh_interval': args.neg_refresh_interval,
            'acc': metrics['acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        row.update(split_stats)
        rows.append(row)
        _write_sanity_row(table_name, row)
        print('SANITY v3 result:', json.dumps(row, ensure_ascii=False))

    def eval_node(method, node_embeds, *, internal_path='none', epoch=0, labels_for_eval=None, target_edges_used=True, **flags):
        labels_for_eval = labels if labels_for_eval is None else labels_for_eval
        metrics = evaluate_fewshot_prototype(
            node_embeds,
            node_embeds,
            labels_for_eval,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            method,
            internal_path,
            epoch,
            'SharedPrototypeEval',
            metrics,
            split_labels=labels_for_eval,
            target_edges_used=target_edges_used,
            **flags,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    eval_node('RawFeaturePrototype', raw_feature, target_edges_used=False)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    projection = torch.randn(raw_feature.size(1), args.unify_dim, generator=gen, device=device) / max(1.0, raw_feature.size(1) ** 0.5)
    random_projected = raw_feature @ projection
    eval_node('RandomProjectionFeaturePrototype', random_projected, target_edges_used=False)

    for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
        eval_node(
            'FeatureOnlyPrototype_LabelShuffle',
            feature,
            labels_for_eval=shuffled_labels,
            label_shuffle=True,
            label_shuffle_seed=shuffle_seed,
            target_edges_used=False,
        )

    perm = torch.randperm(feature.size(0), device=device)
    feature_permuted = feature[perm]

    if target_adj.is_sparse:
        dense_target_adj = target_adj.to_dense()
    else:
        dense_target_adj = target_adj
    degree_raw = dense_target_adj.bool().float().sum(dim=1, keepdim=True)
    degree = (degree_raw - degree_raw.mean(dim=0, keepdim=True)) / degree_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    struct_feats = structural_role_features(target_adj).to(device)
    degree_features = torch.cat([
        degree,
        torch.log1p(degree_raw.clamp_min(0.0)),
        (degree_raw - degree_raw.mean(dim=0, keepdim=True)) / degree_raw.std(dim=0, keepdim=True).clamp_min(1e-6),
    ], dim=1)
    eval_node('DegreeOnlyPrototype', degree_features)
    eval_node('StructuralFeaturePrototype', struct_feats)

    random_model = PrePrompt(
        args.unify_dim,
        args.hid_units,
        'prelu',
        max(1, len(source_domains)),
        args.layers_num,
        0.1,
        type_=args.combinetype,
        backbone=args.backbone,
        alpha=args.alpha,
        ablation=args.ablation_pre,
    ).to(device)
    random_model.eval()
    with torch.no_grad():
        random_h = random_model.gcn(feature, target_adj, sparse, False).squeeze(0)
        random_h_no_edges = random_model.gcn(feature, target_adj_identity, sparse, False).squeeze(0)
        random_h_edge = random_model.gcn(feature, target_adj_edge_permuted, sparse, False).squeeze(0)
        random_h_feature = random_model.gcn(feature_permuted, target_adj, sparse, False).squeeze(0)
    eval_node('RandomEncoderPrototype', random_h)
    eval_node('RandomEncoderPrototype_NoEdges', random_h_no_edges, target_edges_used=False)
    eval_node('RandomEncoderPrototype_EdgePermuted', random_h_edge, edge_permuted=True, target_edges_used=False)
    eval_node('RandomEncoderPrototype_FeaturePermuted', random_h_feature, feature_permuted=True)

    no_pretrain_model = PrePrompt(
        args.unify_dim,
        args.hid_units,
        'prelu',
        max(1, len(source_domains)),
        args.layers_num,
        0.1,
        type_=args.combinetype,
        backbone=args.backbone,
        alpha=args.alpha,
        ablation=args.ablation_pre,
    ).to(device)
    no_pretrain_model.eval()
    with torch.no_grad():
        no_pretrain_h = no_pretrain_model.gcn(feature, target_adj, sparse, False).squeeze(0)
    eval_node('NoPretrainGCNPrototype', no_pretrain_h)
    for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
        eval_node(
            'NoPretrainGCNPrototype_LabelShuffle',
            no_pretrain_h,
            labels_for_eval=shuffled_labels,
            label_shuffle=True,
            label_shuffle_seed=shuffle_seed,
        )

    source_graphs = []
    for name in source_domains:
        src_feature, _, _, src_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, src_feature, src_adj))

    for epoch_count in args.sanity_epochs:
        start = time.time()
        usp_model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        usp_head = USPPretrainingHead(
            args.hid_units,
            readout=args.readout,
            k=args.subgraph_hop,
            temperature=args.usp_temperature,
            align_type=args.align_type if args.use_structure_token else 'none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        opt = torch.optim.Adam(list(usp_model.parameters()) + list(usp_head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(epoch_count):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts = [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = usp_model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = usp_model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = usp_head(
                    usp_model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + args.usp_lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    ego_exclusion_rows.extend(_offset_ego_exclusion_rows(
                        _csr_mask_to_row_lists(out['subgraph_mask'].bool()),
                        node_offset,
                        device,
                    ))
                node_offset += out['node_embeds'].size(0)
            all_h = torch.cat(h_parts, dim=0)
            all_g = torch.cat(g_parts, dim=0)
            domain_ids = torch.cat(domain_parts, dim=0)
            if args.negative_sampling == 'domain_balanced':
                neg_idx, neg_stats = domain_balanced_negative_indices(
                    domain_ids,
                    num_negatives=args.num_negatives,
                    intra_neg_ratio=args.intra_neg_ratio,
                    ego_exclusion_rows=ego_exclusion_rows if ego_exclusion_rows else None,
                    exclude_scope=args.neg_exclude_scope,
                )
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
            loss_ns, _ = sampled_info_nce_indexed(
                all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
            )
            align_loss = torch.tensor(0.0, device=device)
            total_loss = total_loss + args.usp_lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            total_loss.backward()
            opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }
        usp_model.eval()
        usp_head.eval()
        with torch.no_grad():
            target_h = usp_head.encode(usp_model.gcn, feature, target_adj, sparse=sparse)
            target_g, _ = usp_head.subgraph_embeddings(target_h, target_adj, k=args.subgraph_hop)
        usp_metrics = evaluate_fewshot_prototype(
            target_h,
            target_g,
            labels,
            support_idx,
            query_idx,
            query_mode=args.query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit('USPFull', 'USP', epoch_count, 'SharedPrototypeEval', usp_metrics, time.time() - start, loss_stats)
        for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
            usp_shuffle_metrics = evaluate_fewshot_prototype(
                target_h,
                target_g,
                shuffled_labels,
                support_idx,
                query_idx,
                query_mode=args.query_mode,
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit(
                'USPFull_LabelShuffle',
                'USP',
                epoch_count,
                'SharedPrototypeEval',
                usp_shuffle_metrics,
                time.time() - start,
                loss_stats,
                label_shuffle=True,
                label_shuffle_seed=shuffle_seed,
                split_labels=shuffled_labels,
            )

    _write_rows_csv(table_name, rows)
    _write_sanity_summary(summary_name, rows)
    return table_name


def run_formal_batch1(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        domain_balanced_negative_indices,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_minimal_formal_batch1.csv')
    summary_name = os.path.join(result_dir, 'usp_minimal_formal_batch1_summary.csv')
    if args.overwrite_sanity_table:
        for path in [table_name, summary_name]:
            if os.path.exists(path):
                os.remove(path)

    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    target_feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device).long()
    base_query_idx = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    query_idx = base_query_idx

    source_graphs = []
    for name in source_domains:
        src_feature, _, _, src_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, src_feature, src_adj))

    _validate_neg_exclude_scope(args, source_graphs)

    rows = []

    def emit(
        method,
        internal_path,
        shot,
        seed,
        metrics,
        support_idx,
        support_labels,
        query_idx,
        train_time=0.0,
        loss_stats=None,
    ):
        loss_stats = loss_stats or {}
        split_stats = _split_integrity_stats(labels, support_idx, support_labels, query_idx, labels)
        row = {
            'method': method,
            'internal_pretrain_path': internal_path,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'subgraph_hop': args.subgraph_hop if method.startswith('USP') else None,
            'readout': args.readout if method.startswith('USP') else None,
            'query_mode': args.query_mode if method.startswith('USP') else 'node',
            'prototype': args.prototype,
            'negative_sampling': args.negative_sampling if method.startswith('USP') else None,
            'neg_exclude_scope': args.neg_exclude_scope if method.startswith('USP') else None,
            'neg_refresh_interval': args.neg_refresh_interval if method.startswith('USP') else None,
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'acc': metrics['acc'],
            'balanced_acc': metrics['balanced_acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        row.update(split_stats)
        rows.append(row)
        _write_sanity_row(table_name, row)
        print('FORMAL batch1 result:', json.dumps(row, ensure_ascii=False))

    def eval_embeddings(method, internal_path, node_embeds, subgraph_embeds, query_mode, shot, seed, train_time=0.0, loss_stats=None):
        support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, seed, device)
        query_idx = base_query_idx[~torch.isin(base_query_idx, support_idx)]
        metrics = evaluate_fewshot_prototype(
            node_embeds,
            subgraph_embeds,
            labels,
            support_idx,
            query_idx,
            query_mode=query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(method, internal_path, shot, seed, metrics, support_idx, support_labels, query_idx, train_time, loss_stats)

    def train_samgpt(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        start = time.time()
        model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        graphcl_losses = []
        for _ in range(args.formal_epoch):
            opt.zero_grad()
            total = torch.tensor(0.0, device=device)
            losses = []
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                seq = torch.stack([
                    src_feature,
                    src_feature[torch.randperm(src_feature.size(0), device=device)],
                    src_feature.detach(),
                    src_feature.detach(),
                ])
                aug1 = _usp_edge_dropout(src_adj, args.drop_percent)
                aug2 = _usp_edge_dropout(src_adj, args.drop_percent)
                adjs = torch.stack([src_adj.to_dense(), aug1.to_dense(), aug2.to_dense()])
                lbl = torch.cat([torch.ones(1, src_feature.size(0)), torch.zeros(1, src_feature.size(0))], dim=1).to(device)
                logits = next(
                    model.compute_prelogits_GRAPHCL(
                        [model.feature_prompt_layers[graph_id]],
                        [model.structure_prompt_layers[graph_id]],
                        [seq],
                        [adjs],
                        sparse=False,
                        msk=None,
                        samp_bias1=None,
                        samp_bias2=None,
                    )
                )
                loss = model.loss(logits, lbl)
                total = total + loss
                losses.append(float(loss.detach()))
            total.backward()
            opt.step()
            graphcl_losses = losses
        model.eval()
        with torch.no_grad():
            h = model.gcn(target_feature, target_adj, sparse, False).squeeze(0)
        return h, {
            'loss_stat_type': 'final_epoch_mean',
            'L_ns_final_epoch_mean': None,
            'L_ss_final_epoch_mean': float(np.mean(graphcl_losses)) if graphcl_losses else None,
            'L_align_final_epoch_mean': None,
        }, time.time() - start

    def train_usp(seed, method, lambda_ns, lambda_ss):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        start = time.time()
        model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        head = USPPretrainingHead(
            args.hid_units,
            readout=args.readout,
            k=args.subgraph_hop,
            temperature=args.usp_temperature,
            align_type='none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(args.formal_epoch):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts = [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = head(
                    model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    ego_exclusion_rows.extend(_offset_ego_exclusion_rows(
                        _csr_mask_to_row_lists(out['subgraph_mask'].bool()),
                        node_offset,
                        device,
                    ))
                node_offset += out['node_embeds'].size(0)
            all_h = torch.cat(h_parts, dim=0)
            all_g = torch.cat(g_parts, dim=0)
            domain_ids = torch.cat(domain_parts, dim=0)
            if args.negative_sampling == 'domain_balanced':
                neg_idx, neg_stats = domain_balanced_negative_indices(
                    domain_ids,
                    num_negatives=args.num_negatives,
                    intra_neg_ratio=args.intra_neg_ratio,
                    ego_exclusion_rows=ego_exclusion_rows if ego_exclusion_rows else None,
                    exclude_scope=args.neg_exclude_scope,
                )
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
            loss_ns, _ = sampled_info_nce_indexed(
                all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
            )
            align_loss = torch.tensor(0.0, device=device)
            total_loss = total_loss + lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            if total_loss.requires_grad and float(total_loss.detach()) != 0.0:
                total_loss.backward()
                opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }
        model.eval()
        head.eval()
        with torch.no_grad():
            h = head.encode(model.gcn, target_feature, target_adj, sparse=sparse)
            g, _ = head.subgraph_embeddings(h, target_adj, k=args.subgraph_hop)
        return h, g, loss_stats, time.time() - start

    for seed in args.formal_seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        feature_embeds = target_feature
        no_pretrain_model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            max(1, len(source_graphs)),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        no_pretrain_model.eval()
        with torch.no_grad():
            no_pretrain_h = no_pretrain_model.gcn(target_feature, target_adj, sparse, False).squeeze(0)

        sam_h, sam_loss_stats, sam_time = train_samgpt(seed)
        usp_h, usp_g, usp_loss_stats, usp_time = train_usp(seed, 'USPFull', args.usp_lambda_ns, args.usp_lambda_ss)
        nosub_h, nosub_g, nosub_loss_stats, nosub_time = train_usp(seed, 'USPNoSubgraphObjective', 0.0, 0.0)

        for shot in args.shots:
            eval_embeddings('FeatureOnlyPrototype', 'none', feature_embeds, feature_embeds, 'node', shot, seed)
            eval_embeddings('NoPretrainGCNPrototype', 'none', no_pretrain_h, no_pretrain_h, 'node', shot, seed)
            eval_embeddings('SAMGPTPrototypeEval', 'GRAPHCL', sam_h, sam_h, 'node', shot, seed, sam_time, sam_loss_stats)
            eval_embeddings('USPFull', 'USP', usp_h, usp_g, args.query_mode, shot, seed, usp_time, usp_loss_stats)
            eval_embeddings(
                'USPNoSubgraphObjective',
                'USP',
                nosub_h,
                nosub_g,
                args.query_mode,
                shot,
                seed,
                nosub_time,
                nosub_loss_stats,
            )

    _write_rows_csv(table_name, rows)
    _write_formal_batch1_summary(summary_name, rows)
    return table_name


def run_query_objective_ablation(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        domain_balanced_negative_indices,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_query_objective_ablation.csv')
    summary_name = os.path.join(result_dir, 'usp_query_objective_ablation_summary.csv')
    if args.overwrite_sanity_table:
        for path in [table_name, summary_name]:
            if os.path.exists(path):
                os.remove(path)

    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    target_feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device).long()
    base_query_idx = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    query_idx = base_query_idx

    source_graphs = []
    for name in source_domains:
        src_feature, _, _, src_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, src_feature, src_adj))

    rows = []

    def train_usp_variant(seed, method, lambda_ns, lambda_ss):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        start = time.time()
        model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            len(source_graphs),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        head = USPPretrainingHead(
            args.hid_units,
            readout=args.readout,
            k=args.subgraph_hop,
            temperature=args.usp_temperature,
            align_type='none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        encoder_snapshot = _snapshot_named_params(model.gcn)
        readout_snapshot = _snapshot_named_params(head.readout)
        prompt_snapshot = _snapshot_named_params(head.readout, lambda name, _: 'prompt' in name)
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(args.formal_epoch):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts = [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = head(
                    model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    ego_exclusion_rows.extend(_offset_ego_exclusion_rows(
                        _csr_mask_to_row_lists(out['subgraph_mask'].bool()),
                        node_offset,
                        device,
                    ))
                node_offset += out['node_embeds'].size(0)
            all_h = torch.cat(h_parts, dim=0)
            all_g = torch.cat(g_parts, dim=0)
            domain_ids = torch.cat(domain_parts, dim=0)
            if args.negative_sampling == 'domain_balanced':
                neg_idx, neg_stats = domain_balanced_negative_indices(
                    domain_ids,
                    num_negatives=args.num_negatives,
                    intra_neg_ratio=args.intra_neg_ratio,
                    ego_exclusion_rows=ego_exclusion_rows if ego_exclusion_rows else None,
                    exclude_scope=args.neg_exclude_scope,
                )
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
            loss_ns, _ = sampled_info_nce_indexed(
                all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
            )
            align_loss = torch.tensor(0.0, device=device)
            total_loss = total_loss + lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            if total_loss.requires_grad and float(total_loss.detach()) != 0.0:
                total_loss.backward()
                opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }

        encoder_delta = _delta_norm_from_snapshot(model.gcn, encoder_snapshot)
        readout_delta = _delta_norm_from_snapshot(head.readout, readout_snapshot)
        prompt_delta = _delta_norm_from_snapshot(head.readout, prompt_snapshot, lambda name, _: 'prompt' in name)
        model.eval()
        head.eval()
        with torch.no_grad():
            h = head.encode(model.gcn, target_feature, target_adj, sparse=sparse)
            g, _ = head.subgraph_embeddings(h, target_adj, k=args.subgraph_hop)
        loss_stats.update({
            'encoder_param_delta_norm': encoder_delta,
            'readout_param_delta_norm': readout_delta,
            'prompt_param_delta_norm': prompt_delta,
        })
        return h, g, loss_stats, time.time() - start

    def emit(method, objective_active, lambda_ns, lambda_ss, query_mode, shot, seed, metrics, support_idx, support_labels, query_idx, train_time, loss_stats):
        split_stats = _split_integrity_stats(labels, support_idx, support_labels, query_idx, labels)
        row = {
            'method': method,
            'objective_active': objective_active,
            'usp_lambda_ns': lambda_ns,
            'usp_lambda_ss': lambda_ss,
            'query_mode': query_mode,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'subgraph_hop': args.subgraph_hop,
            'max_neighbors': args.max_neighbors,
            'readout': args.readout,
            'csr_readout': args.csr_readout,
            'prototype': args.prototype,
            'negative_sampling': args.negative_sampling,
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'neg_exclude_scope': loss_stats.get('neg_exclude_scope'),
            'neg_refresh_interval': loss_stats.get('neg_refresh_interval'),
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'encoder_param_delta_norm': loss_stats.get('encoder_param_delta_norm'),
            'readout_param_delta_norm': loss_stats.get('readout_param_delta_norm'),
            'prompt_param_delta_norm': loss_stats.get('prompt_param_delta_norm'),
            'acc': metrics['acc'],
            'balanced_acc': metrics['balanced_acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        row.update({
            'support_query_overlap': split_stats['support_query_overlap'],
            'split_integrity_passed': split_stats['split_integrity_passed'],
        })
        rows.append(row)
        _write_sanity_row(table_name, row)
        print('QUERY objective result:', json.dumps(row, ensure_ascii=False))

    variants = [
        ('USPFull', True, 1.0, 1.0),
        ('USPNoSubgraphObjective', False, 0.0, 0.0),
    ]
    query_modes = ['node', 'subgraph', 'hybrid']
    for seed in args.formal_seeds:
        trained = {}
        for method, objective_active, lambda_ns, lambda_ss in variants:
            h, g, loss_stats, train_time = train_usp_variant(seed, method, lambda_ns, lambda_ss)
            trained[method] = (h, g, loss_stats, train_time, objective_active, lambda_ns, lambda_ss)
        for shot in args.shots:
            support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, seed, device)
            query_idx = base_query_idx[~torch.isin(base_query_idx, support_idx)]
            for method, (h, g, loss_stats, train_time, objective_active, lambda_ns, lambda_ss) in trained.items():
                for query_mode in query_modes:
                    metrics = evaluate_fewshot_prototype(
                        h,
                        g,
                        labels,
                        support_idx,
                        query_idx,
                        query_mode=query_mode,
                        prototype_type=args.prototype,
                        metric='cosine',
                    )
                    emit(
                        method,
                        objective_active,
                        lambda_ns,
                        lambda_ss,
                        query_mode,
                        shot,
                        seed,
                        metrics,
                        support_idx,
                        support_labels,
                        query_idx,
                        train_time,
                        loss_stats,
                    )

    _write_rows_csv(table_name, rows)
    _write_query_objective_summary(summary_name, rows)
    return table_name


if args.query_objective_ablation:
    output_path = run_query_objective_ablation(args, parent_directory, sparse=True)
    print(f'Query/objective ablation saved to {output_path}')
    sys.exit(0)


if args.formal_batch1:
    output_path = run_formal_batch1(args, parent_directory, sparse=True)
    print(f'Formal batch1 saved to {output_path}')
    sys.exit(0)


if args.sanity_calibration:
    if args.sanity_version == 'v3':
        output_path = run_sanity_calibration_v3(args, parent_directory, sparse=True)
    else:
        output_path = run_sanity_calibration(args, parent_directory, sparse=True)
    print(f'Sanity calibration saved to {output_path}')
    sys.exit(0)


if args.original_pretrain_method == 'USP':
    output_path = run_usp_minimal(args, parent_directory, current_dir, sparse=True)
    print(f'USP minimal logs saved to {output_path}')
    sys.exit(0)

# ------------------- 训练/模型超参 -------------------
nb_epochs = args.epochs
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
device = runtime_device

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
negetive_sample = torch.tensor(0.0)

print(pretrain_dataset_names)

# 预训练“图”的数量：
# 原实现把 graphId 融入计数（多图数据集/多图选择的情形）
num_pretrain_dataset_num = len(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1

# 为每个预训练数据集创建 DataLoader（本项目多数是单图数据集，但写法比较通用）
data_root = os.path.join(parent_directory, 'data')
pretrain_loaders = [DataLoader(load_dataset(dataset, path=data_root)) for dataset in pretrain_dataset_names]

unify_dim = args.unify_dim

# 日志与输出文件
logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 从 graphId 生成字符串（用于区分多图设置）
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

# 用一堆超参组合出实验名，作为 checkpoint/csv 的文件名
    set_name = (
        f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
        f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
        f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}_'
        f'k{args.subgraph_hop}_mn{args.max_neighbors}_{args.readout}_{args.query_mode}_{args.prototype}_'
        f'csr{int(args.csr_readout)}'
    )
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
).to(device)

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))
except:
    # ------------------- 预训练数据准备（特征/邻接/增强/负采样）-------------------
    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        # 只训练指定的图 id
        if (step + 1) not in target_graph_id:
            continue

        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            # feature/adj 缓存：避免每次运行都重复 process_tu 与 PCA
            rebuild_feature_cache = not (
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            )
            if not rebuild_feature_cache:
                cached_feature = torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                rebuild_feature_cache = cached_feature.size(1) != unify_dim

            if rebuild_feature_cache:
                feature, adj = process.process_tu(data, data.x.shape[1])
                # PCA 压缩/统一到 unify_dim 维度
                feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')

            feature, adj = (
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'),
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_adj.pt'),
            )

            # ----------- GRAPHCL 方式：需要两份增强视图与对比标签 -----------
            if args.pretrain_method == 'GRAPHCL':
                rebuild_aug_cache = not (
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                )
                if not rebuild_aug_cache:
                    cached_aug_feature = torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    rebuild_aug_cache = cached_aug_feature[0].shape[-1] != unify_dim

                if rebuild_aug_cache:
                    aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent)
                    torch.save(aug_feature, f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    torch.save(aug_adj, f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                    torch.save(lbl, f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')

                aug_feature, aug_adj, lbl = (
                    torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt'),
                    torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt'),
                    torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt'),
                )
                aug_features.append(aug_feature)
                aug_adjs.append(aug_adj)
                lbls.append(lbl)

            # ----------- splitLP：每个数据集单独做负采样缓存 -----------
            if args.pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
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

    # ------------------- 优化器 & 搬到 GPU -------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.to(device)

        # features: Tensor -> cuda
        features = [tensors.to(device) for tensors in features]
        # adjs: scipy sparse -> torch sparse -> cuda
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).to(device) if sparse else torch.FloatTensor(adj.todense()).to(device)
            for adj in adjs
        ]
        lbls = [tensors.to(device) for tensors in lbls]
        negetive_samples = [tensors.to(device) for tensors in negetive_samples]

        # LP 情况下 negetive_samples 可能为空，此时使用合并图上的 negetive_sample
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.to(device)
        aug_adjs = [tensors.to(device) for tensors in aug_adjs]
        aug_features = [tensors.to(device) for tensors in aug_features]

    # ------------------- 预训练循环（early stopping）-------------------
    for epoch in range(nb_epochs):
        # 这里每轮都重设 seed，保证增强/负采样等随机过程在 epoch 间可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        loss = 0
        model.train()
        optimiser.zero_grad()

        # GRAPHCL: 传入 (aug_features, aug_adjs, lbls)
        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)

        # LP / splitLP: 传入负采样 samples
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss = model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)

        loss.backward()
        optimiser.step()

        print('Loss:[{:.8f}]'.format(loss))
        # 基于 loss 的 early stopping，并保存最优 checkpoint
        if loss < best:
            firstbest = 1
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))


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

downstream_dataset = load_dataset(args.dataset, path=data_root)
print(downstream_dataset)
downstream_loader = DataLoader(downstream_dataset)
for data in downstream_loader:
    print(data)

    features, adj = process.process_tu(data, data.x.shape[1])
    print('process done')
    features = torch.FloatTensor(pca_compression(features, k=unify_dim)).to(device)

    # 归一化 + 自环
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_sp = adj

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
        test_subgraph = process.build_subgraph(
            adj.todense().A,
            torch.tensor(idx_test),
            False,
            subgraph_hop=args.subgraph_hop,
            max_neighbors=args.max_neighbors,
        )
        test_index = test_subgraph['idx'].to(device)
        test_batch = test_subgraph['batch'].to(device)
    else:
        from downprompt import downprompt

    # 把邻接转 torch sparse（供 GNN 前向）
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(device)
    else:
        adj = torch.FloatTensor(adj.todense()).to(device)

# 加载最优预训练模型并计算 embedding
print(f'loading model from {save_name}')
model.load_state_dict(load_state_dict_trusted(save_name))
model = model.to(device)

eval_start_msg = (
    f'[EVAL] start downstream_task={args.downstream_task} '
    f'query_mode={args.query_mode} subgraph_hop={args.subgraph_hop} '
    f'max_neighbors={args.max_neighbors} readout={args.readout} prototype={args.prototype}'
)
print(eval_start_msg)
logging.info(eval_start_msg)

# model.embed 返回 embedding；embeds[0, idx] 是节点 idx 的表示
embeds, _ = model.embed(features, adj, sparse, None, LP)

# 下游学习率列表（当前只尝试一个值）
downstreamlrlist = [0.001]

# 测试节点的 embedding（仅用于调试/备用）
test_embs = embeds[0, idx_test]

# ------------------- few-shot 下游评测循环 -------------------
for downstreamlr in downstreamlrlist:
    test_lbls = labels[idx_test].to(device)
    accs = []
    macrof = []
    microf = []
    print('-' * 100)

    for shotnum in range(1, shot_num + 1):
        tot = torch.zeros(1).to(device)
        accs = []
        macrof = []
        microf = []

        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum", shotnum)

        # 100 个 few-shot split（与 generate_fewshot.py 对应）
        split_ids = range(args.split_start, args.split_start + args.num_splits)
        for i in tqdm(split_ids):
            split_log_msg = (
                f'[EVAL] split={i} shot={shotnum} '
                f'downstream_task={args.downstream_task} query_mode={args.query_mode} '
                f'subgraph_hop={args.subgraph_hop} max_neighbors={args.max_neighbors}'
            )
            print(split_log_msg)
            logging.info(split_log_msg)

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
            ).to(device)

            log.train()

            # 读取 few-shot 训练集（graph 任务还需读取 batch）
            if args.downstream_task == 'graph':
                idx_train = torch.load(
                    os.path.join(parent_directory, "data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    ))
                ).type(torch.long).to(device)

                batch_train = torch.load(
                    os.path.join(parent_directory, "data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    ))
                ).type(torch.long).to(device)

                lbls_train = torch.load(
                    os.path.join(parent_directory, "data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    ))
                ).type(torch.long).squeeze().to(device)

            else:
                idx_train = torch.load(
                    os.path.join(parent_directory, "data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i))
                ).type(torch.long).to(device)

                lbls_train = torch.load(
                    os.path.join(parent_directory, "data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i))
                ).type(torch.long).squeeze().to(device)

            # 取训练节点 embedding（这里只是取出来，后面真正训练时 downprompt 内部可能还会重新编码）
            pretrain_embs = embeds[0, idx_train]

            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.to(device)
            best = 1e9
            best_acc = torch.zeros(1).to(device)

            # 在 few-shot 训练集上训练分类头，最多 400 step，并用 loss 做 early stopping
            for _ in range(400):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().to(device)
                else:
                    logits = log(
                        features,
                        adj,
                        sparse,
                        model.gcn,
                        idx_train,
                        lbls_train,
                        1,
                        args,
                        adj_sp,
                        sparse,
                    ).float().to(device)

                loss = xent(logits, lbls_train)
                if loss < best:
                    best = loss
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
                logits = log(features, adj, sparse, model.gcn, idx_test, None, 0, args, adj_sp, sparse)

            preds = torch.argmax(logits, dim=1).to(device)
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
        print('Average accuracy:[{:.4f}]'.format(tot.item() / max(1, args.num_splits)))
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
        if args.original_pretrain_method == 'SAMGPT':
            table_name = os.path.join(result_dir, 'usp_minimal_table.csv')
            row = {
                'method': 'SAMGPT',
                'internal_pretrain_path': args.pretrain_method,
                'setting': 'source_only',
                'source_domains': pretrain_dataset_names,
                'target_domain': args.dataset,
                'shot': shotnum,
                'seed': args.seed,
                'adaptation_setting': 'transductive',
                'target_unlabeled_used': True,
                'target_edges_used': True,
                'label_shuffle': False,
                'feature_permuted': False,
                'subgraph_hop': None,
                'readout': None,
                'query_mode': None,
                'prototype': None,
                'negative_sampling': None,
                'num_intra_neg': None,
                'num_inter_neg': None,
                'num_negatives_requested': None,
                'num_negatives_effective': None,
                'intra_neg_ratio_requested': None,
                'intra_neg_ratio_effective': None,
                'negative_fallback_reason': None,
                'negative_sampling_effective': None,
                'use_structure_token': args.ablation_pre in {'all', 'st'},
                'align_type': 'samgpt_structure_token' if args.ablation_pre in {'all', 'st'} else 'none',
                'L_ns_final_epoch_mean': None,
                'L_ss_final_epoch_mean': None,
                'L_align_final_epoch_mean': None,
                'loss_stat_type': None,
                'train_time': None,
                'acc': acc_mean,
                'macro_f1': macrof_mean,
                'micro_f1': microf_mean,
                'margin': None,
                'compactness': None,
                'separation': None,
            }
            write_header = not os.path.exists(table_name)
            with open(table_name, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            with open(os.path.join(result_dir, 'usp_minimal_runs.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
