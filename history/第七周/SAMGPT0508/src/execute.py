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
from tqdm import tqdm
from sklearn.cluster import KMeans
import hashlib

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
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or LP or splitLP')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help="target graph's id in one dataset")
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--graphcl_batch_size', type=int, default=1024, help='mini-batch anchor nodes per domain for GRAPHCL; <=0 uses all selected nodes')
parser.add_argument('--graphcl_negative_ratio', type=int, default=1, help='number of other selected nodes used as negatives per anchor in mini-batch GRAPHCL')
parser.add_argument('--graphcl_stream_domains', type=int, default=1, help='backpropagate GRAPHCL one domain at a time to reduce peak memory')
parser.add_argument('--graphcl_loss_mode', type=str, default='nce', choices=['bce', 'nce', 'bce_plus_nce'], help='mutually exclusive GRAPHCL objective: original BCE, replacement node InfoNCE, or legacy additive BCE+NCE')
parser.add_argument('--graphcl_nce_lambda', type=float, default=0.5, help='weight for node-level in-batch InfoNCE using other selected nodes as negatives')
parser.add_argument('--graphcl_nce_temperature', type=float, default=0.2, help='temperature for node-level in-batch InfoNCE')
parser.add_argument('--cluster_quality_weight', type=int, default=1, help='use cluster-centrality soft weights in node-level InfoNCE')
parser.add_argument('--cluster_keep_ratio', type=float, default=1.0, help='per-domain ratio of high-quality clustered nodes kept for pretraining')
parser.add_argument('--cluster_num', type=int, default=10, help='KMeans clusters per domain for high-quality node selection')
parser.add_argument('--cluster_min_nodes', type=int, default=1, help='minimum selected nodes per domain after cluster filtering')
parser.add_argument('--cluster_kmeans_batch_size', type=int, default=10000, help='use MiniBatchKMeans above this per-domain node count to reduce CPU memory')
parser.add_argument('--cluster_visualize', type=int, default=1, help='save before/after cluster filtering PNGs for each pretrain domain')
parser.add_argument('--debug_log_steps', type=int, default=1, help='print debug logs for major pipeline steps')
parser.add_argument('--cluster_balance_mode', type=str, default='fixed', choices=['fixed', 'ratio'], help='fixed uses the same selected-node quota per domain; ratio keeps the same per-domain keep ratio')
parser.add_argument('--cluster_nodes_per_domain', type=int, default=0, help='explicit fixed selected-node quota per domain; <=0 uses the smallest ratio-based quota')
parser.add_argument('--denoise_anchor_weight', type=int, default=0, help='use cluster-margin reliability as soft anchor weights')
parser.add_argument('--denoise_negative_mask', type=int, default=0, help='mask unsafe/similar negatives in node-level NCE')
parser.add_argument('--denoise_sim_threshold', type=float, default=0.6, help='feature-space cosine threshold above which negatives are treated as unsafe')
parser.add_argument('--denoise_min_negatives', type=int, default=16, help='minimum safe negatives per anchor after denoising mask fallback')
parser.add_argument('--denoise_weight_alpha', type=float, default=1.0, help='strength of reliability weights around domain-normalized mean 1')
parser.add_argument('--denoise_log_interval', type=int, default=50, help='log denoising mask/weight stats every N epochs')
parser.add_argument('--denoise_visualize', type=int, default=1, help='save cluster-guided denoising reliability PNGs')
parser.add_argument('--dynamic_denoise', type=int, default=1, help='enable training-time dynamic denoising from online embedding consistency')
parser.add_argument('--dynamic_denoise_ema', type=float, default=0.95, help='EMA momentum for per-node dynamic reliability')
parser.add_argument('--dynamic_denoise_mix', type=float, default=0.5, help='mixing weight between static cluster reliability and dynamic EMA reliability')
parser.add_argument('--dynamic_denoise_sim_threshold', type=float, default=0.7, help='embedding cosine threshold above which negatives are masked as likely false negatives')
parser.add_argument('--debug_print_full_sim', type=int, default=0, help='print full dynamic-denoise similarity table regardless of batch size (use for tiny debug runs)')
parser.add_argument('--debug_save_full_sim', type=int, default=0, help='save full dynamic-denoise similarity/mask/index-map to files (recommended for large all-node runs)')
parser.add_argument('--debug_sim_subset_index_file', type=str, default='', help='CSV containing matrix_idx/idx values; save only this similarity submatrix instead of a huge full matrix')
parser.add_argument('--nce_diag_interval', type=int, default=0, help='log false-negative diagnostics for node-level NCE every N epochs; <=0 disables')
parser.add_argument('--nce_diag_max_nodes', type=int, default=1024, help='maximum batch nodes per domain used by NCE diagnostics')
parser.add_argument('--nce_diag_sim_threshold', type=float, default=0.6, help='similarity threshold used to count likely false-negative pairs')
parser.add_argument('--nce_diag_visualize', type=int, default=1, help='save NCE diagnostic histograms when diagnostics are enabled')
parser.add_argument('--domain_align_lambda', type=float, default=0.0, help='weight for online cross-domain prototype NegCLIP hard-negative loss')
parser.add_argument('--domain_align_clusters', type=int, default=0, help='shared online prototype count; <=0 reuses cluster_num')
parser.add_argument('--domain_align_temperature', type=float, default=0.2, help='temperature for online prototype alignment')
parser.add_argument('--domain_align_balance_weight', type=float, default=0.05, help='prototype usage-balance regularizer weight')
parser.add_argument('--skip_pretrain', type=int, default=0, help='try to use trained models')
parser.add_argument('--downstream_eval_interval', type=int, default=1000, help='evaluate downstream validation every N pretrain epochs; <=0 falls back to loss early stopping')
parser.add_argument('--downstream_eval_patience', type=int, default=3, help='pretrain early-stop patience counted in downstream validation checks; <=0 disables this stop')
parser.add_argument('--downstream_eval_splits', type=int, default=100, help='few-shot splits used for downstream validation during pretraining')
parser.add_argument('--downstream_eval_train_steps', type=int, default=400, help='training steps for each downstream validation head')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--max_epochs', type=int, default=10000, help='maximum number of training epochs')

args = parser.parse_args()

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
# 当脚本位于 src/execute.py 时，项目根是 src 的上一级；
# 当脚本直接位于项目根时，项目根就是当前目录。
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
sys.path.append(parent_directory)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

# ------------------- 训练/模型超参 -------------------
nb_epochs = args.max_epochs
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
negetive_sample = torch.tensor(0.0)

print(pretrain_dataset_names)

# 预训练“图”的数量：
# 原实现把 graphId 融入计数（多图数据集/多图选择的情形）
num_pretrain_dataset_num = len(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1

# 为每个预训练数据集创建 DataLoader（本项目多数是单图数据集，但写法比较通用）
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

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
    f'gclbs{args.graphcl_batch_size}_gclneg{args.graphcl_negative_ratio}_'
    f'gclmode{args.graphcl_loss_mode}_'
    f'gclnce{args.graphcl_nce_lambda}_'
    f'gclstream{args.graphcl_stream_domains}_'
    f'clkeep{args.cluster_keep_ratio}_clnum{args.cluster_num}_'
    f'clbal{args.cluster_balance_mode}_clquota{args.cluster_nodes_per_domain}_'
    f'denaw{args.denoise_anchor_weight}_denmask{args.denoise_negative_mask}_'
    f'denth{args.denoise_sim_threshold}_'
    f'dynden{args.dynamic_denoise}_dyndm{args.dynamic_denoise_mix}_dyndt{args.dynamic_denoise_sim_threshold}_'
    f'dalign{args.domain_align_lambda}_dcl{args.domain_align_clusters}'
)


def shorten_experiment_name(name, max_len=180):
    """Keep filenames/dirs under OS limits while preserving uniqueness."""
    if len(name) <= max_len:
        return name
    digest = hashlib.md5(name.encode('utf-8')).hexdigest()[:12]
    keep = max(32, max_len - (len(digest) + 5))
    return f'{name[:keep]}__h{digest}'


set_name = shorten_experiment_name(set_name, max_len=180)
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG if args.debug_log_steps else logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
logging.getLogger().addHandler(console_handler)


def debug_step(message):
    """Print and persist high-level debug logs for each pipeline step."""
    if args.debug_log_steps:
        print(f'[DEBUG] {message}', flush=True)
    logging.debug(message)


def build_denoise_batch_signals(metadata_list, batch_indices, device):
    """Return per-domain anchor weights, safe-negative masks, and log stats."""
    batch_weights = []
    negative_masks = []
    stats = []
    for domain_id, (metadata, batch_idx) in enumerate(zip(metadata_list, batch_indices)):
        batch_cpu = batch_idx.detach().cpu().numpy()
        weights_np = metadata['quality_weights'][batch_cpu]
        weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
        batch_weights.append(weights)

        mask = None
        safe_ratio = 1.0
        safe_mean = float(max(int(batch_idx.numel()) - 1, 0))
        if args.denoise_negative_mask:
            labels = torch.as_tensor(metadata['labels'][batch_cpu], dtype=torch.long, device=device)
            feature_norm = torch.as_tensor(metadata['feature_norm'][batch_cpu], dtype=torch.float32, device=device)
            sim = torch.matmul(feature_norm, feature_norm.t())
            node_num = int(batch_idx.numel())
            eye = torch.eye(node_num, dtype=torch.bool, device=device)
            diff_cluster = labels.unsqueeze(0) != labels.unsqueeze(1)
            mask = diff_cluster & (sim < args.denoise_sim_threshold) & (~eye)

            min_neg = max(0, min(args.denoise_min_negatives, node_num - 1))
            if min_neg > 0 and node_num > 1:
                counts = mask.sum(dim=1)
                need = counts < min_neg
                if bool(need.any().item()):
                    fallback_scores = sim.masked_fill(eye, float('inf'))
                    _, fallback_idx = torch.topk(fallback_scores, k=min_neg, dim=1, largest=False)
                    row_idx = torch.arange(node_num, device=device).unsqueeze(1).expand_as(fallback_idx)
                    fallback_mask = torch.zeros_like(mask)
                    fallback_mask[row_idx, fallback_idx] = True
                    mask = mask | (fallback_mask & need.unsqueeze(1))

            denom = max(node_num * max(node_num - 1, 0), 1)
            safe_ratio = float(mask.sum().detach().item()) / float(denom)
            safe_mean = float(mask.sum(dim=1).float().mean().detach().item()) if node_num > 0 else 0.0
        negative_masks.append(mask)
        stats.append(
            {
                'domain_id': domain_id,
                'batch': int(batch_idx.numel()),
                'weight_mean': float(weights.mean().detach().item()) if weights.numel() else 0.0,
                'weight_min': float(weights.min().detach().item()) if weights.numel() else 0.0,
                'weight_max': float(weights.max().detach().item()) if weights.numel() else 0.0,
                'safe_ratio': safe_ratio,
                'safe_mean': safe_mean,
            }
        )
    return batch_weights, negative_masks, stats


def init_dynamic_denoise_state(node_pools):
    state = []
    for pool in node_pools:
        node_count = int(np.max(pool)) + 1 if len(pool) > 0 else 0
        state.append(
            {
                'ema_reliability': np.ones(node_count, dtype=np.float32),
            }
        )
    return state


@torch.no_grad()
def build_dynamic_denoise_batch_signals(
    model,
    batch_indices,
    aug_features,
    aug_adjs,
    sparse,
    device,
    state,
    metadata_list=None,
    epoch=None,
    output_dir=None,
):
    batch_weights = []
    negative_masks = []
    stats = []
    z1_all = []
    z2_all = []
    sizes = []
    global_domain_ids = []
    global_node_ids = []
    global_cluster_labels = []
    for domain_id, batch_idx in enumerate(batch_indices):
        domain_feature = aug_features[domain_id].to(device)
        domain_adj = aug_adjs[domain_id].to(device)
        z1, z2 = model.compute_graphcl_node_embeddings(domain_id, domain_feature, domain_adj, sparse, node_idx=batch_idx)
        pos_sim = (z1 * z2).sum(dim=-1).clamp(-1.0, 1.0)
        cur = state[domain_id]['ema_reliability']
        idx_np = batch_idx.detach().cpu().numpy()
        prev = torch.as_tensor(cur[idx_np], dtype=torch.float32, device=device)
        updated = args.dynamic_denoise_ema * prev + (1.0 - args.dynamic_denoise_ema) * ((pos_sim + 1.0) * 0.5)
        cur[idx_np] = updated.detach().cpu().numpy()

        batch_weights.append(updated.clamp_min(1e-3))
        z1_all.append(z1)
        z2_all.append(z2)
        sizes.append(int(batch_idx.numel()))
        node_ids_np = batch_idx.detach().cpu().numpy().astype(np.int64)
        global_node_ids.extend(node_ids_np.tolist())
        global_domain_ids.extend([int(domain_id)] * int(batch_idx.numel()))
        if metadata_list is not None and domain_id < len(metadata_list) and metadata_list[domain_id] is not None:
            labels_np = metadata_list[domain_id]['labels'][node_ids_np].astype(np.int64)
            global_cluster_labels.extend(labels_np.tolist())
        else:
            global_cluster_labels.extend([-1] * int(batch_idx.numel()))
        stats.append(
            {
                'domain_id': domain_id,
                'pos_sim_mean': float(pos_sim.mean().item()) if pos_sim.numel() else 0.0,
                'dyn_weight_mean': float(updated.mean().item()) if updated.numel() else 0.0,
                'dyn_weight_min': float(updated.min().item()) if updated.numel() else 0.0,
                'dyn_weight_max': float(updated.max().item()) if updated.numel() else 0.0,
            }
        )

    if len(z1_all) == 0:
        return batch_weights, negative_masks, None, stats

    z1_cat = torch.cat(z1_all, dim=0)
    z2_cat = torch.cat(z2_all, dim=0)
    z_cat = F.normalize(0.5 * (z1_cat + z2_cat), dim=-1)
    sim = torch.matmul(z_cat, z_cat.t())
    total_nodes = sim.shape[0]
    if args.debug_sim_subset_index_file and output_dir is not None and epoch is not None:
        os.makedirs(output_dir, exist_ok=True)
        subset_indices = []
        with open(args.debug_sim_subset_index_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f'Empty debug_sim_subset_index_file: {args.debug_sim_subset_index_file}')
            idx_key = 'matrix_idx' if 'matrix_idx' in reader.fieldnames else 'idx'
            for row in reader:
                subset_indices.append(int(row[idx_key]))
        if not subset_indices:
            raise ValueError(f'No indices found in debug_sim_subset_index_file: {args.debug_sim_subset_index_file}')
        bad = [idx for idx in subset_indices if idx < 0 or idx >= total_nodes]
        if bad:
            raise ValueError(f'Subset indices out of range for sim matrix size {total_nodes}: {bad[:10]}')
        subset_tensor = torch.as_tensor(subset_indices, dtype=torch.long, device=sim.device)
        sim_subset = sim.index_select(0, subset_tensor).index_select(1, subset_tensor).detach().cpu().numpy()
        subset_stem = os.path.splitext(os.path.basename(args.debug_sim_subset_index_file))[0]
        sim_subset_csv = os.path.join(output_dir, f'epoch_{int(epoch):05d}_sim_subset_{subset_stem}.csv')
        map_subset_csv = os.path.join(output_dir, f'epoch_{int(epoch):05d}_index_map_subset_{subset_stem}.csv')
        np.savetxt(sim_subset_csv, sim_subset, delimiter=',', fmt='%.6f')
        with open(map_subset_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sub_pos', 'matrix_idx', 'domain', 'node_id', 'cluster'])
            for sub_pos, matrix_idx in enumerate(subset_indices):
                writer.writerow([
                    sub_pos,
                    matrix_idx,
                    global_domain_ids[matrix_idx],
                    global_node_ids[matrix_idx],
                    global_cluster_labels[matrix_idx],
                ])
        debug_step(
            f'dynamic-denoise subset sim saved: sim={sim_subset_csv}, index_map={map_subset_csv}, '
            f'subset_size={len(subset_indices)} (epoch={epoch})'
        )
    if args.debug_save_full_sim and output_dir is not None and epoch is not None:
        os.makedirs(output_dir, exist_ok=True)
        sim_cpu = sim.detach().cpu().numpy()
        mask_pre = (sim < args.dynamic_denoise_sim_threshold) & (~torch.eye(total_nodes, dtype=torch.bool, device=device))
        mask_pre_cpu = mask_pre.detach().cpu().numpy().astype(np.int8)
        sim_csv = os.path.join(output_dir, f'epoch_{int(epoch):05d}_sim.csv')
        mask_csv = os.path.join(output_dir, f'epoch_{int(epoch):05d}_mask_pre_fallback.csv')
        np.savetxt(sim_csv, sim_cpu, delimiter=',', fmt='%.6f')
        np.savetxt(mask_csv, mask_pre_cpu, delimiter=',', fmt='%d')
        map_path = os.path.join(output_dir, f'epoch_{int(epoch):05d}_index_map.csv')
        with open(map_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'domain', 'node_id', 'cluster'])
            for i, (d, n, c) in enumerate(zip(global_domain_ids, global_node_ids, global_cluster_labels)):
                writer.writerow([i, d, n, c])
        debug_step(
            f'dynamic-denoise full matrices saved: sim={sim_csv}, mask={mask_csv}, '
            f'index_map={map_path} (epoch={epoch})'
        )
    if args.debug_log_steps and (args.debug_print_full_sim or total_nodes <= 32):
        sim_cpu = sim.detach().cpu().numpy()
        sim_rows = np.array2string(sim_cpu, precision=3, suppress_small=True, max_line_width=200)
        # debug_step(f'dynamic-denoise sim-matrix shape={sim_cpu.shape}\n{sim_rows}')
        neg_ratio = float((sim < 0).sum().item()) / float(max(sim.numel(), 1))
        debug_step(f'dynamic-denoise sim<0 ratio={neg_ratio:.4f} (negative cosine means dissimilar nodes)')
        sim_headers = [f"sim_to_idx{j}" for j in range(total_nodes)]
        header = "idx | domain | node_id | cluster | " + " | ".join(sim_headers)
        rows = [header, "-" * len(header)]
        for i, (d, n, c) in enumerate(zip(global_domain_ids, global_node_ids, global_cluster_labels)):
            sim_cols = " | ".join(f"{v:>10.3f}" for v in sim_cpu[i].tolist())
            rows.append(f"{i:>3} | {d:>6} | {n:>7} | {c:>7} | {sim_cols}")
        debug_step("dynamic-denoise index-map table\n" + "\n".join(rows))
    eye = torch.eye(total_nodes, dtype=torch.bool, device=device)
    global_mask = (sim < args.dynamic_denoise_sim_threshold) & (~eye)
    if args.debug_log_steps and (args.debug_print_full_sim or total_nodes <= 32):
        mask_cpu = global_mask.detach().cpu().numpy().astype(np.int32)
        mask_rows = np.array2string(mask_cpu, separator=',', max_line_width=200)
        # debug_step(f'dynamic-denoise keep-neg-mask(1=keep,0=drop) shape={mask_cpu.shape}\n{mask_rows}')

    # Keep at least min_negatives per anchor after dynamic masking.
    min_neg = max(0, min(int(args.denoise_min_negatives), total_nodes - 1))
    if min_neg > 0 and total_nodes > 1:
        counts = global_mask.sum(dim=1)
        need = counts < min_neg
        if bool(need.any().item()):
            fallback_scores = sim.masked_fill(eye, float('inf'))
            _, fallback_idx = torch.topk(fallback_scores, k=min_neg, dim=1, largest=False)
            row_idx = torch.arange(total_nodes, device=device).unsqueeze(1).expand_as(fallback_idx)
            fallback_mask = torch.zeros_like(global_mask)
            fallback_mask[row_idx, fallback_idx] = True
            global_mask = global_mask | (fallback_mask & need.unsqueeze(1))

    # Split global mask into domain-local blocks for streaming mode compatibility.
    offset = 0
    for size in sizes:
        sub = global_mask[offset : offset + size, offset : offset + size]
        negative_masks.append(sub)
        offset += size

    drop_ratio = 1.0 - float(global_mask.sum().item()) / float(max(total_nodes * max(total_nodes - 1, 1), 1))
    for item in stats:
        item['dynamic_drop_ratio'] = drop_ratio
    return batch_weights, negative_masks, global_mask, stats


def _safe_tensor_mean(tensor, default=0.0):
    if tensor is None or tensor.numel() == 0:
        return float(default)
    return float(tensor.float().mean().detach().item())


def _safe_tensor_quantile(tensor, q, default=0.0):
    if tensor is None or tensor.numel() == 0:
        return float(default)
    return float(torch.quantile(tensor.float(), q).detach().item())


def _save_nce_diag_plot(feature_neg, embed_neg, pos_sim, same_cluster_neg, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import matplotlib

        matplotlib.use('Agg')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        debug_step(f'nce diagnostic visualization skipped: matplotlib unavailable ({exc})')
        return None

    def _to_np(tensor):
        if tensor is None or tensor.numel() == 0:
            return np.asarray([], dtype=np.float32)
        return tensor.detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].hist(_to_np(feature_neg), bins=50, alpha=0.85, color='steelblue')
    axes[0].set_title('feature negative sim')
    axes[0].axvline(args.nce_diag_sim_threshold, color='crimson', linestyle='--', linewidth=1)

    axes[1].hist(_to_np(embed_neg), bins=50, alpha=0.75, label='all neg', color='darkorange')
    same_np = _to_np(same_cluster_neg)
    if same_np.size:
        axes[1].hist(same_np, bins=50, alpha=0.55, label='same cluster', color='seagreen')
    axes[1].set_title('embedding negative sim')
    axes[1].legend(loc='best', fontsize=8)

    axes[2].hist(_to_np(pos_sim), bins=40, alpha=0.85, color='mediumpurple')
    axes[2].set_title('positive sim')
    fig.suptitle(title)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def run_nce_noise_diagnostics(model, metadata_list, batch_indices, aug_features, aug_adjs, sparse, epoch, device, output_dir=None):
    """Measure likely false negatives without changing the training objective."""
    if metadata_list is None or args.nce_diag_interval <= 0:
        return []
    stats = []
    max_nodes = max(2, int(args.nce_diag_max_nodes))
    sim_threshold = float(args.nce_diag_sim_threshold)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for domain_id, (metadata, batch_idx) in enumerate(zip(metadata_list, batch_indices)):
            node_num = int(batch_idx.numel())
            if node_num <= 1:
                continue

            if node_num > max_nodes:
                select = torch.randperm(node_num, device=batch_idx.device)[:max_nodes]
                diag_idx = batch_idx[select]
            else:
                diag_idx = batch_idx
            diag_num = int(diag_idx.numel())
            if diag_num <= 1:
                continue

            diag_cpu = diag_idx.detach().cpu().numpy()
            labels = torch.as_tensor(metadata['labels'][diag_cpu], dtype=torch.long, device=device)
            quality = torch.as_tensor(metadata['quality_weights'][diag_cpu], dtype=torch.float32, device=device)
            feature_norm = torch.as_tensor(metadata['feature_norm'][diag_cpu], dtype=torch.float32, device=device)
            feature_sim = torch.matmul(feature_norm, feature_norm.t())
            eye = torch.eye(diag_num, dtype=torch.bool, device=device)
            offdiag = ~eye
            same_cluster = (labels.unsqueeze(0) == labels.unsqueeze(1)) & offdiag
            diff_cluster = (labels.unsqueeze(0) != labels.unsqueeze(1)) & offdiag
            high_sim = (feature_sim >= sim_threshold) & offdiag

            domain_feature = aug_features[domain_id].to(device)
            domain_adj = aug_adjs[domain_id].to(device)
            z1, z2 = model.compute_graphcl_node_embeddings(
                domain_id,
                domain_feature,
                domain_adj,
                sparse,
                node_idx=diag_idx,
            )
            embed_sim = torch.matmul(z1, z2.t())
            pos_sim = torch.diagonal(embed_sim)
            feature_neg = feature_sim[offdiag]
            embed_neg = embed_sim[offdiag]
            same_cluster_feature_neg = feature_sim[same_cluster]
            same_cluster_embed_neg = embed_sim[same_cluster]
            diff_cluster_embed_neg = embed_sim[diff_cluster]

            denom = float(max(diag_num * (diag_num - 1), 1))
            likely_fn = high_sim | same_cluster
            high_same = high_sim & same_cluster
            stat = {
                'domain_id': domain_id,
                'batch': diag_num,
                'quality_mean': _safe_tensor_mean(quality),
                'quality_low_ratio': float((quality < 1.0).float().mean().detach().item()),
                'feature_neg_mean': _safe_tensor_mean(feature_neg),
                'feature_neg_p90': _safe_tensor_quantile(feature_neg, 0.9),
                'high_sim_neg_ratio': float(high_sim.sum().detach().item()) / denom,
                'same_cluster_neg_ratio': float(same_cluster.sum().detach().item()) / denom,
                'same_cluster_high_sim_ratio': float(high_same.sum().detach().item()) / denom,
                'likely_fn_ratio': float(likely_fn.sum().detach().item()) / denom,
                'pos_sim_mean': _safe_tensor_mean(pos_sim),
                'embed_neg_mean': _safe_tensor_mean(embed_neg),
                'embed_neg_p90': _safe_tensor_quantile(embed_neg, 0.9),
                'same_cluster_embed_sim': _safe_tensor_mean(same_cluster_embed_neg),
                'diff_cluster_embed_sim': _safe_tensor_mean(diff_cluster_embed_neg),
                'same_cluster_feature_sim': _safe_tensor_mean(same_cluster_feature_neg),
            }
            stats.append(stat)

            if args.nce_diag_visualize and output_dir:
                domain_name = pretrain_dataset_names[domain_id] if domain_id < len(pretrain_dataset_names) else f'domain_{domain_id}'
                safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(domain_name))
                _save_nce_diag_plot(
                    feature_neg,
                    embed_neg,
                    pos_sim,
                    same_cluster_embed_neg,
                    os.path.join(output_dir, f'epoch_{epoch:05d}_{domain_id:02d}_{safe_name}_nce_diag.png'),
                    f'epoch {epoch} {domain_name}',
                )

            del domain_feature, domain_adj, z1, z2, feature_sim, embed_sim

    if was_training:
        model.train()
    return stats


def log_nce_noise_diagnostics(epoch, stats):
    if not stats:
        return
    n = float(len(stats))
    mean_likely_fn = sum(item['likely_fn_ratio'] for item in stats) / n
    mean_high_sim = sum(item['high_sim_neg_ratio'] for item in stats) / n
    mean_same_cluster = sum(item['same_cluster_neg_ratio'] for item in stats) / n
    mean_pos = sum(item['pos_sim_mean'] for item in stats) / n
    mean_neg = sum(item['embed_neg_mean'] for item in stats) / n
    mean_same_embed = sum(item['same_cluster_embed_sim'] for item in stats) / n
    mean_diff_embed = sum(item['diff_cluster_embed_sim'] for item in stats) / n
    debug_step(
        'NCE-DIAG epoch={} likely_fn_ratio={:.4f} high_sim_neg={:.4f} same_cluster_neg={:.4f} '
        'pos_sim={:.4f} embed_neg={:.4f} same_embed={:.4f} diff_embed={:.4f} details={}'.format(
            epoch,
            mean_likely_fn,
            mean_high_sim,
            mean_same_cluster,
            mean_pos,
            mean_neg,
            mean_same_embed,
            mean_diff_embed,
            [
                (
                    item['domain_id'],
                    item['batch'],
                    round(item['likely_fn_ratio'], 3),
                    round(item['high_sim_neg_ratio'], 3),
                    round(item['same_cluster_neg_ratio'], 3),
                    round(item['pos_sim_mean'], 3),
                    round(item['same_cluster_embed_sim'], 3),
                    round(item['diff_cluster_embed_sim'], 3),
                )
                for item in stats
            ],
        )
    )


def add_dominant_cluster_center_node(feature, pool, n_clusters=10, random_state=0):
    """Add the closest node to the largest cluster centroid into the pool."""
    feat_np = feature.detach().cpu().numpy() if torch.is_tensor(feature) else np.asarray(feature)
    node_num = feat_np.shape[0]
    if node_num == 0:
        return np.asarray(pool, dtype=np.int64), None

    cluster_num = max(1, min(n_clusters, node_num))
    kmeans = KMeans(n_clusters=cluster_num, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(feat_np)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels, minlength=cluster_num)
    dominant_cluster = int(np.argmax(counts))
    members = np.where(labels == dominant_cluster)[0]
    distances = np.linalg.norm(feat_np[members] - centers[dominant_cluster], axis=1)
    center_node = int(members[np.argmin(distances)])

    pool_arr = np.asarray(pool, dtype=np.int64)
    exists = np.isin(center_node, pool_arr)
    if not exists:
        pool_arr = np.unique(np.concatenate([pool_arr, np.array([center_node], dtype=np.int64)]))

    info = {
        'cluster_id': dominant_cluster,
        'cluster_size': int(counts[dominant_cluster]),
        'center_node': center_node,
        'center_distance': float(np.min(distances)),
        'added': not exists,
    }
    return pool_arr, info


def freeze_module_parameters(module):
    """Temporarily freeze a module while still allowing gradients through inputs."""
    requires_grad_states = [param.requires_grad for param in module.parameters()]
    for param in module.parameters():
        param.requires_grad_(False)
    return requires_grad_states


def restore_module_parameters(module, requires_grad_states):
    for param, requires_grad in zip(module.parameters(), requires_grad_states):
        param.requires_grad_(requires_grad)


def prepare_downstream_data():
    """Load downstream data once on CPU; pretrain-time checks move it to GPU lazily."""
    print('#' * 50)
    print('PreTrain datasets are ', pretrain_dataset_names)
    print('Downastream dataset is ', args.dataset)
    logging.info('#' * 50)
    logging.info('PreTrain datasets are ')
    logging.info(pretrain_dataset_names)
    logging.info('Downastream dataset is ')
    logging.info(args.dataset)

    downstream_dataset = load_dataset(args.dataset)
    print(downstream_dataset)
    downstream_loader = DataLoader(downstream_dataset)
    prepared = None
    for graph_data in downstream_loader:
        print(graph_data)

        downstream_features, downstream_adj = process.process_tu(graph_data, graph_data.x.shape[1])
        print('process done')
        downstream_features = torch.FloatTensor(pca_compression(downstream_features, k=unify_dim))

        downstream_adj = process.normalize_adj(downstream_adj + sp.eye(downstream_adj.shape[0]))
        labels = graph_data.y.type(torch.long).cpu()
        node_num = int(labels.shape[0])

        test_start = max(0, node_num - test_idx_num)
        val_end = test_start
        val_start = max(0, val_end - test_idx_num)
        if val_end <= val_start:
            val_start = 0
            val_end = node_num

        idx_val = torch.arange(val_start, val_end, dtype=torch.long)
        idx_test = torch.arange(test_start, node_num, dtype=torch.long)

        labels_np = labels.numpy()
        nb_classes = len(np.unique(labels_np))
        print('nb_classes', nb_classes)
        print(f'validation nodes: {val_start}-{val_end - 1}, test nodes: {test_start}-{node_num - 1}')

        prepared = {
            'dataset': downstream_dataset,
            'features_cpu': downstream_features,
            'adj_sp': downstream_adj,
            'labels_cpu': labels,
            'idx_val_cpu': idx_val,
            'idx_test_cpu': idx_test,
            'nb_classes': nb_classes,
        }

        if args.downstream_task == 'graph':
            adj_dense = downstream_adj.todense().A
            val_subgraph = process.build_subgraph(adj_dense, idx_val, False)
            test_subgraph = process.build_subgraph(adj_dense, idx_test, False)
            prepared.update(
                {
                    'val_index_cpu': val_subgraph['idx'].type(torch.long),
                    'val_batch_cpu': val_subgraph['batch'].type(torch.long),
                    'test_index_cpu': test_subgraph['idx'].type(torch.long),
                    'test_batch_cpu': test_subgraph['batch'].type(torch.long),
                }
            )

    if prepared is None:
        raise RuntimeError(f'No downstream graph loaded for dataset {args.dataset}')
    return prepared


def move_downstream_data_to_device(downstream_data):
    bundle = {
        'features': downstream_data['features_cpu'].cuda(),
        'labels': downstream_data['labels_cpu'].cuda(),
        'idx_val': downstream_data['idx_val_cpu'].cuda(),
        'idx_test': downstream_data['idx_test_cpu'].cuda(),
        'nb_classes': downstream_data['nb_classes'],
    }
    if sparse:
        bundle['adj'] = process.sparse_mx_to_torch_sparse_tensor(downstream_data['adj_sp']).cuda()
    else:
        bundle['adj'] = torch.FloatTensor(downstream_data['adj_sp'].todense()).cuda()

    if args.downstream_task == 'graph':
        bundle.update(
            {
                'val_index': downstream_data['val_index_cpu'].cuda(),
                'val_batch': downstream_data['val_batch_cpu'].cuda(),
                'test_index': downstream_data['test_index_cpu'].cuda(),
                'test_batch': downstream_data['test_batch_cpu'].cuda(),
            }
        )
    return bundle


def load_fewshot_split(shotnum, split_id):
    dataset_lower = args.dataset.lower()
    data_root = os.path.join(parent_directory, 'data')
    if args.downstream_task == 'graph':
        split_dir = os.path.join(
            data_root,
            f'fewshot_{dataset_lower}_graph',
            f'{shotnum}-shot_{dataset_lower}',
            str(split_id),
        )
        idx_train = torch_load_trusted(os.path.join(split_dir, 'idx.pt')).type(torch.long).cuda()
        batch_train = torch_load_trusted(os.path.join(split_dir, 'batch.pt')).type(torch.long).cuda()
        lbls_train = torch_load_trusted(os.path.join(split_dir, 'labels.pt')).type(torch.long).squeeze().cuda()
        return idx_train, batch_train, lbls_train

    split_dir = os.path.join(
        data_root,
        f'fewshot_{dataset_lower}',
        f'{shotnum}-shot_{dataset_lower}',
        str(split_id),
    )
    idx_train = torch_load_trusted(os.path.join(split_dir, 'idx.pt')).type(torch.long).cuda()
    lbls_train = torch_load_trusted(os.path.join(split_dir, 'labels.pt')).type(torch.long).squeeze().cuda()
    return idx_train, None, lbls_train


def write_downstream_csv_row(result, downstream_data):
    with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, dialect="excel")
        writer.writerow(
            [
                pretrain_dataset_str,
                downstream_data['dataset'],
                f"{result['acc_mean']:.3f}",
                f"{result['acc_std']:.3f}",
                f"{result['microf_mean']:.3f}",
                f"{result['microf_std']:.3f}",
                f"{result['macrof_mean']:.3f}",
                f"{result['macrof_std']:.3f}",
            ]
        )


def run_downstream_evaluation(
    pretrain_model,
    downstream_data,
    eval_target='test',
    split_count=100,
    train_steps=400,
    write_csv=False,
    progress=True,
    phase='final',
):
    """Train downstream heads and report validation/test metrics for current pretrain weights."""
    split_count = max(1, int(split_count))
    train_steps = max(1, int(train_steps))
    downstreamlrlist = [0.001]
    DownPrompt = downprompt_graph if args.downstream_task == 'graph' else downprompt

    was_training = pretrain_model.training
    requires_grad_states = freeze_module_parameters(pretrain_model)
    pretrain_model.eval()
    eval_bundle = move_downstream_data_to_device(downstream_data)
    summaries = []

    try:
        for downstreamlr in downstreamlrlist:
            eval_lbls = eval_bundle['labels'][eval_bundle[f'idx_{eval_target}']]
            print('-' * 100)

            for shotnum in range(1, shot_num + 1):
                accs = []
                macrof = []
                microf = []
                print(f"{phase} {eval_target} shotnum", shotnum)

                split_iter = range(split_count)
                if progress:
                    split_iter = tqdm(split_iter)

                for split_id in split_iter:
                    fea_pretext_weights, str_pretext_weights, combines = pretrain_model.get_weights()
                    combines.append(args.beta)

                    log = DownPrompt(
                        hid_units,
                        eval_bundle['nb_classes'],
                        unify_dim,
                        num_layers_num,
                        fea_pretext_weights,
                        str_pretext_weights,
                        combines,
                        args.combinetype,
                        args.ablation_down,
                    ).cuda()
                    log.train()

                    idx_train, batch_train, lbls_train = load_fewshot_split(shotnum, split_id)
                    opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                    best_head_loss = float('inf')
                    head_cnt_wait = 0

                    for _ in range(train_steps):
                        opt.zero_grad()
                        if args.downstream_task == 'graph':
                            logits = log(
                                eval_bundle['features'],
                                eval_bundle['adj'],
                                sparse,
                                pretrain_model.gcn,
                                idx_train,
                                batch_train,
                                lbls_train,
                                1,
                            ).float().cuda()
                        else:
                            logits = log(
                                eval_bundle['features'],
                                eval_bundle['adj'],
                                sparse,
                                pretrain_model.gcn,
                                idx_train,
                                lbls_train,
                                1,
                            ).float().cuda()

                        loss = xent(logits, lbls_train)
                        loss_value = loss.item()
                        if loss_value < best_head_loss:
                            best_head_loss = loss_value
                            head_cnt_wait = 0
                        else:
                            head_cnt_wait += 1
                        if head_cnt_wait == patience:
                            break

                        loss.backward()
                        opt.step()

                    log.eval()
                    with torch.no_grad():
                        if args.downstream_task == 'graph':
                            logits = log(
                                eval_bundle['features'],
                                eval_bundle['adj'],
                                sparse,
                                pretrain_model.gcn,
                                eval_bundle[f'{eval_target}_index'],
                                eval_bundle[f'{eval_target}_batch'],
                            )
                        else:
                            logits = log(
                                eval_bundle['features'],
                                eval_bundle['adj'],
                                sparse,
                                pretrain_model.gcn,
                                eval_bundle[f'idx_{eval_target}'],
                            )

                    preds = torch.argmax(logits, dim=1).cuda()
                    acc = torch.sum(preds == eval_lbls).float() / eval_lbls.shape[0]

                    preds_cpu = preds.cpu().numpy()
                    eval_lbls_cpu = eval_lbls.cpu().numpy()
                    micro_f1 = f1_score(eval_lbls_cpu, preds_cpu, average='micro')
                    macro_f1 = f1_score(eval_lbls_cpu, preds_cpu, average='macro')

                    microf.append(micro_f1 * 100)
                    macrof.append(macro_f1 * 100)
                    accs.append((acc * 100).item())

                    del log, opt, logits, loss

                accs_tensor = torch.tensor(accs)
                acc_mean = accs_tensor.mean().item()
                acc_std = accs_tensor.std().item()
                microf_mean = sum(microf) / len(microf)
                macrof_mean = sum(macrof) / len(macrof)
                microf_std = torch.std(torch.tensor(microf)).item()
                macrof_std = torch.std(torch.tensor(macrof)).item()
                result = {
                    'shotnum': shotnum,
                    'acc_mean': acc_mean,
                    'acc_std': acc_std,
                    'microf_mean': microf_mean,
                    'microf_std': microf_std,
                    'macrof_mean': macrof_mean,
                    'macrof_std': macrof_std,
                }
                summaries.append(result)

                print('-' * 100)
                print(f'{phase} {eval_target} Average accuracy:[{acc_mean:.4f}]')
                print('Mean:[{:.4f}]'.format(acc_mean))
                print('Std :[{:.4f}]'.format(acc_std))
                print('-' * 100)
                logging.info('-' * 100)
                logging.info('%s %s Mean:[%.4f]', phase, eval_target, acc_mean)
                logging.info('%s %s Std :[%.4f]', phase, eval_target, acc_std)
                logging.info('-' * 100)

                if write_csv:
                    write_downstream_csv_row(result, downstream_data)

        return summaries
    finally:
        restore_module_parameters(pretrain_model, requires_grad_states)
        if was_training:
            pretrain_model.train()
        else:
            pretrain_model.eval()
        del eval_bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mean_downstream_accuracy(results):
    if not results:
        return float('-inf')
    return sum(result['acc_mean'] for result in results) / len(results)


debug_step(f'project_root={parent_directory}, checkpoints={save_dir}, cache={cache_dir}, results={result_dir}')

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
    domain_align_clusters=args.domain_align_clusters if args.domain_align_clusters > 0 else args.cluster_num,
).cuda()

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId
downstream_data = prepare_downstream_data()

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    debug_step(f'try loading pretrained checkpoint: {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))
    debug_step(f'checkpoint loaded successfully: {save_name}')
except (FileNotFoundError, AssertionError) as exc:
    debug_step(f'pretraining required ({exc}); checkpoint path: {save_name}')
    # ------------------- 预训练数据准备（特征/邻接/增强/负采样）-------------------
    for step, datas in enumerate(zip(*pretrain_loaders)):
        debug_step(f'pretrain data step={step}')
        # 只训练指定的图 id
        if (step + 1) not in target_graph_id:
            continue

        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            debug_step(f'prepare dataset={pretrain_dataset_name}: raw feature_dim={data.x.shape[1]} nodes={data.x.shape[0]}')
            # feature/adj 缓存：避免每次运行都重复 process_tu 与 PCA
            if not (
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            ):
                debug_step(f'cache miss for {pretrain_dataset_name}: process_tu + PCA')
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
                if not (
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                ):
                    debug_step(f'cache miss for {pretrain_dataset_name}: build GRAPHCL augmentation drop_percent={drop_percent}')
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
                    debug_step(f'cache miss for {pretrain_dataset_name}: build splitLP negative samples')
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

    # ----------- GRAPHCL cluster filtering: balanced per-domain high-quality node pools -----------
    graphcl_node_pools = None
    graphcl_quality_weights = None
    graphcl_denoise_metadata = None
    graphcl_dynamic_denoise_state = None
    if args.pretrain_method == 'GRAPHCL':
        cluster_vis_dir = os.path.join(result_dir, 'cluster_visualizations', set_name)
        denoise_vis_dir = os.path.join(result_dir, 'denoise_visualizations', set_name)
        nce_diag_vis_dir = os.path.join(result_dir, 'nce_diagnostics', set_name)
        sim_dump_dir = os.path.join(result_dir, 'sim_dumps', set_name)
        debug_step(
            f'GRAPHCL cluster filtering start: keep_ratio={args.cluster_keep_ratio}, '
            f'clusters={args.cluster_num}, balance={args.cluster_balance_mode}, visualize={args.cluster_visualize}'
        )
        graphcl_node_pools = preprompt.build_balanced_domain_node_indices(
            features,
            keep_ratio=args.cluster_keep_ratio,
            n_clusters=args.cluster_num,
            min_nodes=args.cluster_min_nodes,
            random_state=seed,
            balance_mode=args.cluster_balance_mode,
            nodes_per_domain=args.cluster_nodes_per_domain,
            kmeans_batch_size=args.cluster_kmeans_batch_size,
            visualize=bool(args.cluster_visualize),
            visualization_dir=cluster_vis_dir,
            domain_names=pretrain_dataset_names,
        )
        use_all_nodes = all(len(pool) >= feature.shape[0] for pool, feature in zip(graphcl_node_pools, features))
        if args.cluster_visualize and not use_all_nodes:
            debug_step(f'cluster before/after visualizations saved under: {cluster_vis_dir}')
        elif use_all_nodes:
            debug_step('GRAPHCL uses all nodes; cluster filtering and center-node supplementation skipped')
        for idx, (domain_name, pool, feature) in enumerate(zip(pretrain_dataset_names, graphcl_node_pools, features)):
            pool_before = len(pool)
            center_info = None
            if not use_all_nodes:
                updated_pool, center_info = add_dominant_cluster_center_node(
                    feature,
                    pool,
                    n_clusters=args.cluster_num,
                    random_state=seed + idx,
                )
                graphcl_node_pools[idx] = updated_pool
            pool_after = len(graphcl_node_pools[idx])
            print(
                f'GRAPHCL selected nodes for {domain_name}: '
                f'{pool_after}/{feature.shape[0]} '
                f'(mode={args.cluster_balance_mode}, per-domain cluster filtering)'
            )
            if center_info is not None:
                debug_step(
                    'cluster-center '
                    f'domain={domain_name} cluster={center_info["cluster_id"]} '
                    f'size={center_info["cluster_size"]} '
                    f'node={center_info["center_node"]} '
                    f'dist={center_info["center_distance"]:.6f} '
                    f'pool={pool_before}->{pool_after} added={center_info["added"]}'
                )
        if args.cluster_quality_weight:
            graphcl_quality_weights = preprompt.compute_cluster_quality_weights(
                features,
                n_clusters=args.cluster_num,
                random_state=seed,
                kmeans_batch_size=args.cluster_kmeans_batch_size,
            )
            debug_step('GRAPHCL cluster quality weights enabled for node-level NCE')
        if args.denoise_anchor_weight or args.denoise_negative_mask or args.nce_diag_interval > 0:
            graphcl_denoise_metadata = preprompt.compute_cluster_denoising_metadata(
                features,
                n_clusters=args.cluster_num,
                random_state=seed,
                kmeans_batch_size=args.cluster_kmeans_batch_size,
                weight_alpha=args.denoise_weight_alpha,
            )
            debug_step(
                f'cluster-guided denoising enabled: anchor_weight={args.denoise_anchor_weight}, '
                f'negative_mask={args.denoise_negative_mask}, sim_threshold={args.denoise_sim_threshold}, '
                f'min_negatives={args.denoise_min_negatives}, nce_diag_interval={args.nce_diag_interval}'
            )
            for domain_id, (domain_name, feature, metadata) in enumerate(zip(pretrain_dataset_names, features, graphcl_denoise_metadata)):
                debug_step(
                    'denoise metadata domain={} nodes={} clusters={} quality[min={:.4f} mean={:.4f} max={:.4f} std={:.4f}] '
                    'margin[mean={:.4f} std={:.4f}]'.format(
                        domain_name,
                        feature.shape[0],
                        metadata['cluster_num'],
                        metadata.get('quality_min', 1.0),
                        metadata.get('quality_mean', 1.0),
                        metadata.get('quality_max', 1.0),
                        metadata.get('quality_std', 0.0),
                        metadata.get('margin_mean', 0.0),
                        metadata.get('margin_std', 0.0),
                    )
                )
                if args.denoise_visualize:
                    safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(domain_name))
                    preprompt.visualize_denoising_metadata(
                        feature,
                        metadata,
                        output_path=os.path.join(denoise_vis_dir, f'{domain_id:02d}_{safe_name}_denoising.png'),
                        title=str(domain_name),
                        random_state=seed + domain_id,
                    )
            if args.denoise_visualize:
                debug_step(f'denoising visualizations saved under: {denoise_vis_dir}')
            if args.nce_diag_interval > 0 and args.nce_diag_visualize:
                os.makedirs(nce_diag_vis_dir, exist_ok=True)
                debug_step(f'NCE diagnostic visualizations will be saved under: {nce_diag_vis_dir}')
        if args.dynamic_denoise:
            graphcl_dynamic_denoise_state = init_dynamic_denoise_state(graphcl_node_pools)
            debug_step(
                'dynamic denoise enabled: ema={:.3f}, mix={:.3f}, sim_threshold={:.3f}'.format(
                    args.dynamic_denoise_ema,
                    args.dynamic_denoise_mix,
                    args.dynamic_denoise_sim_threshold,
                )
            )

    # ------------------- 优化器 & 搬到 GPU -------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        debug_step('Using CUDA; moving model/tensors according to streaming settings')
        model = model.cuda()

        if args.pretrain_method == 'GRAPHCL' and args.graphcl_stream_domains:
            # Keep large per-domain GRAPHCL tensors on CPU and move only the
            # current domain to GPU inside the training loop. This avoids
            # pinning every augmented graph/view in GPU memory at once.
            pass
        else:
            # features: Tensor -> cuda
            features = [tensors.cuda() for tensors in features]
            # adjs: scipy sparse -> torch sparse -> cuda
            adjs = [
                process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
                for adj in adjs
            ]
            lbls = [tensors.cuda() for tensors in lbls]
            aug_adjs = [tensors.cuda() for tensors in aug_adjs]
            aug_features = [tensors.cuda() for tensors in aug_features]

        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        # LP 情况下 negetive_samples 可能为空，此时使用合并图上的 negetive_sample
        if len(negetive_samples) == 0 and args.pretrain_method != 'GRAPHCL':
            negetive_samples = negetive_sample.cuda()

    # ------------------- 预训练循环（early stopping）-------------------
    eval_interval = args.downstream_eval_interval
    downstream_eval_enabled = eval_interval > 0
    loss_best_name = os.path.join(save_dir, f'{set_name}_loss_best.pkl')
    best_downstream_metric = float('-inf')
    downstream_best_t = -1
    downstream_cnt_wait = 0
    debug_step(
        f'pretraining loop start: nb_epochs={nb_epochs}, loss_patience={patience}, '
        f'downstream_eval_interval={eval_interval}, downstream_eval_patience={args.downstream_eval_patience}'
    )
    for epoch in range(nb_epochs):
        # Use deterministic but epoch-varying seeds. Reusing the same seed here
        # makes mini-batch anchors and negatives identical in every epoch.
        np.random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        torch.cuda.manual_seed(seed + epoch)

        loss = 0
        backward_done = False
        model.train()
        optimiser.zero_grad()

        # GRAPHCL: mini-batch anchors are sampled from balanced per-domain
        # cluster-selected node pools; negatives come from other selected nodes.
        if args.pretrain_method == 'GRAPHCL':
            graphcl_device = next(model.parameters()).device
            graphcl_batch_indices, graphcl_negative_indices = preprompt.sample_graphcl_minibatches(
                graphcl_node_pools,
                args.graphcl_batch_size,
                negative_ratio=args.graphcl_negative_ratio,
                device=graphcl_device,
            )
            graphcl_batch_weights = None
            graphcl_negative_masks = None
            graphcl_global_negative_mask = None
            graphcl_anchor_weights = None
            denoise_stats = None
            if graphcl_quality_weights is not None:
                graphcl_batch_weights = []
                for weights_np, batch_idx in zip(graphcl_quality_weights, graphcl_batch_indices):
                    weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=graphcl_device)
                    graphcl_batch_weights.append(weights_tensor[batch_idx])
            if graphcl_denoise_metadata is not None:
                denoise_weights, graphcl_negative_masks, denoise_stats = build_denoise_batch_signals(
                    graphcl_denoise_metadata,
                    graphcl_batch_indices,
                    graphcl_device,
                )
                if args.denoise_anchor_weight:
                    graphcl_anchor_weights = denoise_weights
                    if graphcl_batch_weights is None:
                        graphcl_batch_weights = denoise_weights
                if not args.denoise_negative_mask:
                    graphcl_negative_masks = None
            dynamic_stats = None
            if args.dynamic_denoise and graphcl_dynamic_denoise_state is not None:
                dyn_weights, dyn_masks, dyn_global_mask, dynamic_stats = build_dynamic_denoise_batch_signals(
                    model,
                    graphcl_batch_indices,
                    aug_features,
                    aug_adjs,
                    sparse,
                    graphcl_device,
                    graphcl_dynamic_denoise_state,
                    metadata_list=graphcl_denoise_metadata,
                    epoch=epoch,
                    output_dir=sim_dump_dir if (args.debug_save_full_sim or args.debug_sim_subset_index_file) else None,
                )
                if graphcl_batch_weights is None:
                    graphcl_batch_weights = dyn_weights
                else:
                    mixed_weights = []
                    mix = float(max(0.0, min(1.0, args.dynamic_denoise_mix)))
                    for st_w, dy_w in zip(graphcl_batch_weights, dyn_weights):
                        mixed_weights.append((1.0 - mix) * st_w + mix * dy_w)
                    graphcl_batch_weights = mixed_weights
                graphcl_anchor_weights = graphcl_batch_weights
                graphcl_negative_masks = dyn_masks
                graphcl_global_negative_mask = dyn_global_mask
            if epoch == 0:
                pool_sizes = [len(pool) for pool in graphcl_node_pools]
                batch_sizes = [int(batch.numel()) for batch in graphcl_batch_indices]
                min_batch = min(batch_sizes) if batch_sizes else 0
                max_batch = max(batch_sizes) if batch_sizes else 0
                full_batch = all(b == p for b, p in zip(batch_sizes, pool_sizes)) if batch_sizes else False
                mode = 'full' if full_batch or args.graphcl_batch_size <= 0 else 'mini-batch'
                debug_step(
                    f'GRAPHCL batching: mode={mode}, requested={args.graphcl_batch_size}, '
                    f'per-domain batch min={min_batch} max={max_batch}, '
                    f'loss_mode={args.graphcl_loss_mode}, nce_lambda={args.graphcl_nce_lambda}'
                )
            if denoise_stats is not None and args.denoise_log_interval > 0 and epoch % args.denoise_log_interval == 0:
                mean_weight = sum(item['weight_mean'] for item in denoise_stats) / len(denoise_stats)
                mean_safe_ratio = sum(item['safe_ratio'] for item in denoise_stats) / len(denoise_stats)
                mean_safe_neg = sum(item['safe_mean'] for item in denoise_stats) / len(denoise_stats)
                debug_step(
                    'denoise epoch={} weight_mean={:.4f} safe_ratio={:.4f} safe_neg/anchor={:.2f} details={}'.format(
                        epoch,
                        mean_weight,
                        mean_safe_ratio,
                        mean_safe_neg,
                        [
                            (
                                item['domain_id'],
                                round(item['weight_min'], 3),
                                round(item['weight_mean'], 3),
                                round(item['weight_max'], 3),
                                round(item['safe_ratio'], 3),
                                round(item['safe_mean'], 1),
                            )
                            for item in denoise_stats
                        ],
                    )
                )
            if dynamic_stats is not None and args.denoise_log_interval > 0 and epoch % args.denoise_log_interval == 0:
                mean_pos_sim = sum(item['pos_sim_mean'] for item in dynamic_stats) / len(dynamic_stats)
                mean_dyn_w = sum(item['dyn_weight_mean'] for item in dynamic_stats) / len(dynamic_stats)
                mean_drop = sum(item['dynamic_drop_ratio'] for item in dynamic_stats) / len(dynamic_stats)
                debug_step(
                    'dynamic-denoise epoch={} pos_sim_mean={:.4f} dyn_weight_mean={:.4f} drop_ratio={:.4f} details={}'.format(
                        epoch,
                        mean_pos_sim,
                        mean_dyn_w,
                        mean_drop,
                        [
                            (
                                item['domain_id'],
                                round(item['pos_sim_mean'], 3),
                                round(item['dyn_weight_min'], 3),
                                round(item['dyn_weight_mean'], 3),
                                round(item['dyn_weight_max'], 3),
                            )
                            for item in dynamic_stats
                        ],
                    )
                )
            if args.nce_diag_interval > 0 and epoch % args.nce_diag_interval == 0:
                nce_diag_stats = run_nce_noise_diagnostics(
                    model,
                    graphcl_denoise_metadata,
                    graphcl_batch_indices,
                    aug_features,
                    aug_adjs,
                    sparse,
                    epoch,
                    graphcl_device,
                    output_dir=nce_diag_vis_dir if args.nce_diag_visualize else None,
                )
                log_nce_noise_diagnostics(epoch, nce_diag_stats)
            if args.graphcl_stream_domains:
                loss = torch.tensor(0.0, device=graphcl_device)
                loss_graphcl = torch.tensor(0.0, device=graphcl_device)
                loss_align = torch.tensor(0.0, device=graphcl_device)
                loss_nce = torch.tensor(0.0, device=graphcl_device)
                domain_count = len(aug_features)
                for domain_id in range(domain_count):
                    domain_feature = aug_features[domain_id].to(graphcl_device)
                    domain_adj = aug_adjs[domain_id].to(graphcl_device)
                    domain_lbl = lbls[domain_id].to(graphcl_device)
                    domain_weights = None if graphcl_batch_weights is None else graphcl_batch_weights[domain_id]
                    domain_anchor_weights = None if graphcl_anchor_weights is None else graphcl_anchor_weights[domain_id]
                    domain_loss, domain_graphcl, domain_align, domain_nce = model.graphcl_domain_loss(
                        domain_id,
                        domain_feature,
                        domain_adj,
                        sparse,
                        None,
                        None,
                        None,
                        domain_lbl,
                        node_idx=graphcl_batch_indices[domain_id],
                        neg_idx=graphcl_negative_indices[domain_id],
                        domain_align_lambda=args.domain_align_lambda,
                        domain_align_temperature=args.domain_align_temperature,
                        domain_align_balance_weight=args.domain_align_balance_weight,
                        graphcl_loss_mode=args.graphcl_loss_mode,
                        graphcl_nce_lambda=args.graphcl_nce_lambda,
                        graphcl_nce_temperature=args.graphcl_nce_temperature,
                        graphcl_sample_weights=domain_weights,
                        graphcl_negative_mask=None if graphcl_negative_masks is None else graphcl_negative_masks[domain_id],
                        graphcl_anchor_weights=domain_anchor_weights,
                        return_parts=True,
                    )
                    (domain_loss / domain_count).backward()
                    loss = loss + domain_loss.detach() / domain_count
                    loss_graphcl = loss_graphcl + domain_graphcl.detach() / domain_count
                    loss_align = loss_align + domain_align.detach() / domain_count
                    loss_nce = loss_nce + domain_nce.detach() / domain_count
                    del domain_loss, domain_feature, domain_adj, domain_lbl
                backward_done = True
            else:
                loss, loss_graphcl, loss_align, loss_nce = model(
                    aug_features,
                    aug_adjs,
                    sparse,
                    None,
                    None,
                    None,
                    lbls,
                    None,
                    graphcl_node_indices=graphcl_batch_indices,
                    graphcl_negative_indices=graphcl_negative_indices,
                    domain_align_lambda=args.domain_align_lambda,
                    domain_align_temperature=args.domain_align_temperature,
                    domain_align_balance_weight=args.domain_align_balance_weight,
                    graphcl_loss_mode=args.graphcl_loss_mode,
                    graphcl_nce_lambda=args.graphcl_nce_lambda,
                    graphcl_nce_temperature=args.graphcl_nce_temperature,
                    graphcl_sample_weights=graphcl_batch_weights,
                    graphcl_negative_masks=graphcl_negative_masks,
                    graphcl_global_negative_mask=graphcl_global_negative_mask,
                    graphcl_anchor_weights=graphcl_anchor_weights,
                    return_parts=True,
                )

        # LP / splitLP: 传入负采样 samples
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss = model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)

        if not backward_done:
            loss.backward()
        optimiser.step()

        if args.pretrain_method == 'GRAPHCL':
            debug_step(
                'epoch={} Loss:[{:.8f}] graphcl={:.8f} align={:.8f} nce={:.8f}'.format(
                    epoch,
                    loss,
                    loss_graphcl,
                    loss_align,
                    loss_nce,
                )
            )
        else:
            debug_step('epoch={} Loss:[{:.8f}]'.format(epoch, loss))
        # 保留一份 loss 最优模型作诊断；主 checkpoint 由下游验证指标决定。
        if loss < best:
            firstbest = 1
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), loss_best_name)
        else:
            cnt_wait += 1

        should_eval_downstream = downstream_eval_enabled and ((epoch + 1) % eval_interval == 0 or epoch == nb_epochs - 1)
        if should_eval_downstream:
            debug_step(f'downstream validation start at epoch={epoch + 1}')
            val_results = run_downstream_evaluation(
                model,
                downstream_data,
                eval_target='val',
                split_count=args.downstream_eval_splits,
                train_steps=args.downstream_eval_train_steps,
                write_csv=False,
                progress=True,
                phase=f'pretrain_epoch_{epoch + 1}',
            )
            val_metric = mean_downstream_accuracy(val_results)
            debug_step(
                f'downstream validation epoch={epoch + 1} acc_mean={val_metric:.4f} '
                f'best={best_downstream_metric:.4f}'
            )
            if val_metric > best_downstream_metric:
                best_downstream_metric = val_metric
                downstream_best_t = epoch
                downstream_cnt_wait = 0
                torch.save(model.state_dict(), save_name)
                debug_step(f'new downstream-best checkpoint saved: epoch={epoch + 1}, metric={val_metric:.4f}')
            else:
                downstream_cnt_wait += 1

            if args.downstream_eval_patience > 0 and downstream_cnt_wait >= args.downstream_eval_patience:
                print('Early stopping by downstream validation!')
                debug_step(
                    f'downstream validation early stop: best_epoch={downstream_best_t + 1}, '
                    f'best_acc={best_downstream_metric:.4f}'
                )
                break

        if not downstream_eval_enabled and cnt_wait == patience:
            if os.path.exists(loss_best_name):
                model.load_state_dict(load_state_dict_trusted(loss_best_name))
            torch.save(model.state_dict(), save_name)
            print('Early stopping by pretrain loss!')
            break
        # print('Loading {}th epoch'.format(best_t))

    if downstream_eval_enabled and downstream_best_t < 0:
        torch.save(model.state_dict(), save_name)
        debug_step('no downstream validation checkpoint was produced; saved final pretrain model')
    elif not downstream_eval_enabled and not os.path.exists(save_name):
        if os.path.exists(loss_best_name):
            model.load_state_dict(load_state_dict_trusted(loss_best_name))
        torch.save(model.state_dict(), save_name)


# ------------------- 下游测试评测（使用下游验证最佳 checkpoint）-------------------
debug_step(f'loading downstream-best/pretrained model for final test: {save_name}')
if not os.path.exists(save_name):
    raise FileNotFoundError(f'No checkpoint exists at {save_name}. Set --skip_pretrain 0 or check the checkpoint naming arguments.')
model.load_state_dict(load_state_dict_trusted(save_name))
model = model.cuda()

run_downstream_evaluation(
    model,
    downstream_data,
    eval_target='test',
    split_count=100,
    train_steps=args.downstream_eval_train_steps,
    write_csv=True,
    progress=True,
    phase='final',
)
