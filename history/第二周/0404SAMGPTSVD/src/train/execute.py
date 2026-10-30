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
import json
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


def debug_trace(event, **kwargs):
    """Emit a compact progress line for long-running pipeline stages."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if kwargs:
        details = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        print(f'[trace] {ts} {event}: {details}')
    else:
        print(f'[trace] {ts} {event}')

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
parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='max grad norm; <=0 disables grad clipping')
parser.add_argument('--lr_scheduler_patience', type=int, default=20, help='ReduceLROnPlateau patience on training loss')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='ReduceLROnPlateau decay factor')
parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-5, help='ReduceLROnPlateau minimum learning rate')
parser.add_argument('--early_stop_min_delta', type=float, default=1e-4, help='minimum loss improvement to reset early stopping wait')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
# Stage2 过滤与结构映射设置
parser.add_argument('--max_subgraph_nodes', type=int, default=64, help='unified max subgraph nodes used across all stages; 0 means no cap')
parser.add_argument('--stage2_top_ratio', type=float, default=0.3, help='top ratio selected by learnability rank in each graph')
parser.add_argument('--context_outputs_root', type=str, default='', help='root dir of unified context outputs')
parser.add_argument('--result_subdir', type=str, default='', help='subdir under result/ for this run; default uses timestamp')
parser.add_argument('--flow_check_only', type=int, default=0, help='1: run pipeline sanity checks and exit before optimization')
parser.add_argument('--flow_report_name', type=str, default='flow_check_report.json', help='output json report filename for flow sanity checks')
# Stage1 自动构建参数
parser.add_argument('--stage1_num_clusters', type=int, default=8, help='number of clusters for stage1 auto-prepare')
parser.add_argument('--stage1_cluster_method', type=str, default='gmm', choices=['gmm', 'kmeans'], help='stage1 clustering method')
parser.add_argument('--stage1_subgraph_hop_n', type=int, default=2, help='n-hop neighborhood depth for stage1 subgraph construction')
parser.add_argument('--stage1_encoding_mode', type=str, default='raw9', choices=['raw9', 'decorrelated7'], help='feature encoding mode used by stage1 clustering')
parser.add_argument('--stage1_use_log1p', type=int, default=1, help='1: log1p skewed count features before scaling in stage1')
parser.add_argument('--stage1_remove_selfloops', type=int, default=1, help='1: remove self-loops in stage1; 0: keep self-loops')
# Stage2 自动构建参数
parser.add_argument('--stage2_top_m_per_cluster', type=int, default=5, help='representative pool size per cluster for stage2 auto-prepare')
parser.add_argument('--stage2_alpha', type=float, default=1.0, help='stage2 novelty weight')
parser.add_argument('--stage2_beta', type=float, default=1.0, help='stage2 coverage weight')
parser.add_argument('--stage2_gamma', type=float, default=1.0, help='stage2 redundancy weight')
parser.add_argument('--global_pool_svd_rank', type=int, default=64, help='rank used by SVD low-rank similarity payload in global pool cache')
# EGRAPHCL 专用参数
parser.add_argument('--egraphcl_tau', type=float, default=0.85, help='fixed online soft-positive threshold for EGRAPHCL (hyperparameter, no epoch scheduling)')
parser.add_argument('--egraphcl_temperature', type=float, default=0.2, help='sample-level contrastive temperature for EGRAPHCL')
parser.add_argument('--egraphcl_pair_chunk_size', type=int, default=256, help='anchor chunk size for sampled pair contrastive loss')
parser.add_argument('--egraphcl_candidate_pool', type=int, default=1024, help='sampled candidate pool size for negatives; <=0 means full pool')
parser.add_argument('--egraphcl_soft_topk', type=int, default=10, help='keep only top-k soft positives per anchor for masking; <=0 means keep all')
parser.add_argument('--egraphcl_max_negatives', type=int, default=512, help='max negatives per anchor after masking; <=0 means keep all')
parser.add_argument('--egraphcl_batch_size', type=int, default=128, help='mini-batch size for sample-level EGRAPHCL negative contrast')
parser.add_argument('--egraphcl_mask_source', type=str, default='structure', choices=['structure', 'feature'], help='legacy compatibility arg; ignored in single-branch mode (mask is built from encoded sample similarity)')
parser.add_argument('--egraphcl_min_negatives_per_anchor', type=int, default=1, help='minimum negatives per anchor kept after soft mask (feature branch)')
parser.add_argument('--egraphcl_structure_min_negatives_per_anchor', type=int, default=8, help='minimum negatives per anchor kept after soft mask (structure branch)')
parser.add_argument('--egraphcl_max_mask_ratio', type=float, default=0.95, help='upper bound of masked pair ratio per anchor in online mask')
parser.add_argument('--egraphcl_curve_log_every', type=int, default=0, help='save similarity curve every N epochs; <=0 disables curve export')
parser.add_argument('--egraphcl_curve_bins', type=int, default=200, help='histogram bins for similarity curve')
parser.add_argument('--egraphcl_curve_chunk_rows', type=int, default=512, help='row chunk size when computing similarity histogram')
parser.add_argument('--egraphcl_curve_save_plot', type=int, default=0, help='1 to save PNG plots for similarity curves; 0 saves summary JSON only')
parser.add_argument('--global_subgraph_pool', type=int, default=1, help='1: unified cross-dataset subgraph pool for GRAPHCL/EGRAPHCL')
parser.add_argument('--force_rebuild_global_pool', type=int, default=0, help='1: rebuild merged global subgraph pool cache')
parser.add_argument('--sample_progress_every', type=int, default=1000, help='log interval for sample progress; <=0 disables periodic logs (keeps first/last)')
parser.add_argument('--egraphcl_mask_log_every', type=int, default=5, help='log interval for egraphcl mask stats; <=0 disables periodic logs (keeps first few epochs)')
parser.add_argument('--epoch_log_every', type=int, default=5, help='log interval for training loss/epoch; <=0 disables periodic logs (keeps first few epochs)')
parser.add_argument('--log_cuda_peak_mem', type=int, default=1, help='1: log CUDA peak memory per epoch when CUDA is available')
args = parser.parse_args()

args.max_subgraph_nodes = int(args.max_subgraph_nodes)
if args.max_subgraph_nodes < 0:
    raise ValueError('max_subgraph_nodes must be >= 0 (0 means no cap)')

if args.pretrain_method == 'EGRAPHCL' and args.ablation_pre != 'all':
    print(
        '[context] EGRAPHCL requires dual-prompt alignment after sample mapping; '
        f'force ablation_pre from {args.ablation_pre} to all.'
    )
    args.ablation_pre = 'all'

if args.pretrain_method == 'EGRAPHCL' and int(args.global_subgraph_pool) != 1:
    raise ValueError('EGRAPHCL now only supports global_subgraph_pool=1 (sample-similarity mask mode).')

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
from models.stage1_cluster import fit_clusters
from models.stage2_learnability_model import FEATURE_KEYS, compute_learnability
from layers.stage2_pool_builder import build_representative_pool
from layers.stage2_output_writer import write_stage2_outputs
from tools.stage_metrics import evaluate_stage1_artifacts, evaluate_stage2_artifacts, export_stage1_selected_pca_same_axes
from tools.egraphcl_similarity_viz import dump_similarity_curves

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


def _count_soft_mask_pairs(mask_tensor):
    """Count soft-positive masked pairs from BCE mask's negative half."""
    if mask_tensor is None:
        return 0, 0
    neg_mask = mask_tensor[:, mask_tensor.shape[1] // 2:]
    masked = int((neg_mask <= 0).sum().item())
    total = int(neg_mask.numel())
    return masked, total


def _should_log_progress(current, total, every):
    if current == 1 or current == total:
        return True
    if int(every) <= 0:
        return False
    return (int(current) % int(every)) == 0


def run_flow_sanity_checks(
    *,
    args,
    result_dir,
    stage1_outputs_root,
    stage2_outputs_root,
    aug_features,
    aug_adjs,
    lbls,
    bce_masks,
    sample_dataset_ids,
    selected_samples,
):
    """Validate end-to-end pipeline invariants before long training."""

    checks = []

    def add_check(name, ok, detail):
        checks.append({'name': str(name), 'ok': bool(ok), 'detail': str(detail)})
        status = 'PASS' if ok else 'FAIL'
        print(f'[flow-check] {status} {name}: {detail}')

    if args.pretrain_method in ('GRAPHCL', 'EGRAPHCL'):
        n_aug = len(aug_features)
        add_check('non_empty_aug_views', n_aug > 0, f'aug_features={n_aug}')
        add_check(
            'aug_list_length_consistency',
            n_aug == len(aug_adjs) == len(lbls) == len(bce_masks),
            f'aug_features={len(aug_features)}, aug_adjs={len(aug_adjs)}, lbls={len(lbls)}, bce_masks={len(bce_masks)}',
        )

    if sample_dataset_ids is not None:
        add_check(
            'sample_dataset_ids_alignment',
            len(sample_dataset_ids) == len(aug_features),
            f'sample_dataset_ids={len(sample_dataset_ids)}, aug_features={len(aug_features)}',
        )

    if int(args.global_subgraph_pool) == 1:
        stage1_metric_path = os.path.join(stage1_outputs_root, 'global_merged', 'stage1_metrics.json')
        stage2_metric_path = os.path.join(stage2_outputs_root, 'global_merged', 'stage2_metrics.json')
        add_check('stage1_metrics_exist', os.path.exists(stage1_metric_path), stage1_metric_path)
        add_check('stage2_metrics_exist', os.path.exists(stage2_metric_path), stage2_metric_path)

    if len(selected_samples) > 0:
        ds_unique = sorted({str(s['dataset_name']) for s in selected_samples})
        cl_unique = sorted({int(s['cluster_id']) for s in selected_samples})
        add_check(
            'dataset_coverage',
            len(ds_unique) >= 2,
            f'unique_datasets={len(ds_unique)}, names={ds_unique}',
        )
        add_check(
            'cluster_coverage',
            len(cl_unique) >= 2,
            f'unique_clusters={len(cl_unique)}, ids={cl_unique}',
        )

    if args.pretrain_method == 'EGRAPHCL':
        add_check(
            'online_pair_mode_min_samples',
            len(aug_features) > 1,
            f'aug_features={len(aug_features)}, online pair mask is built in model forward',
        )

    all_ok = all(c['ok'] for c in checks)
    report = {
        'all_ok': bool(all_ok),
        'pretrain_method': str(args.pretrain_method),
        'result_dir': str(result_dir),
        'checks': checks,
    }

    report_path = os.path.join(result_dir, args.flow_report_name)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'[flow-check] saved report: {report_path}')

    return all_ok, report_path


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

        need_build_stage1 = not os.path.exists(stage1_csv)
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
                    cluster_method=args.stage1_cluster_method,
                    encoding_mode=args.stage1_encoding_mode,
                    seed=args.seed,
                    remove_selfloops=bool(args.stage1_remove_selfloops),
                    use_log1p=bool(args.stage1_use_log1p),
                )
            )

        ds_stage2_dir = os.path.join(stage2_root, ds_lower)
        stage2_pt = os.path.join(ds_stage2_dir, 'learnability_scores.pt')
        need_build_stage2 = not os.path.exists(stage2_pt)
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
                    top_p_percent=args.stage2_top_ratio,
                    alpha=args.stage2_alpha,
                    beta=args.stage2_beta,
                    gamma=args.stage2_gamma,
                    dataset_name=ds,
                    data_root=os.path.join(parent_directory, 'data'),
                    max_subgraph_nodes=args.max_subgraph_nodes,
                    subgraph_hop_n=args.stage1_subgraph_hop_n,
                    remove_selfloops=bool(args.stage1_remove_selfloops),
                    pca_dim=unify_dim,
                    write_decoded_subgraphs=True,
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


def _build_svd_similarity_payload(selected_unit: np.ndarray, max_rank: int = 64) -> dict:
    """Build a low-rank payload so we do not materialize an N x N similarity matrix.

    We represent S ~= Z Z^T where Z = U * sqrt(Sigma), from thin SVD of selected_unit.
    """
    if selected_unit.ndim != 2:
        raise ValueError(f"selected_unit must be 2D, got shape={selected_unit.shape}")

    u, s, _ = np.linalg.svd(selected_unit, full_matrices=False)
    k = int(min(max_rank, s.shape[0]))
    u_k = u[:, :k]
    s_k = np.maximum(s[:k], 0.0)
    z = (u_k * np.sqrt(s_k)).astype(np.float32)
    n = int(selected_unit.shape[0])
    return {
        'format': 'svd_gram',
        'z': z,
        'shape': (n, n),
        'rank': int(k),
    }


def _similarity_shape(sim_payload) -> tuple[int, int]:
    if isinstance(sim_payload, np.ndarray):
        if sim_payload.ndim == 2:
            return int(sim_payload.shape[0]), int(sim_payload.shape[1])
        return 0, 0

    if isinstance(sim_payload, dict) and sim_payload.get('format') == 'svd_gram':
        shp = sim_payload.get('shape', (0, 0))
        if isinstance(shp, (list, tuple)) and len(shp) == 2:
            return int(shp[0]), int(shp[1])
    return 0, 0


def _is_valid_similarity_payload(sim_payload, n_selected: int) -> bool:
    if n_selected <= 0:
        return False

    if isinstance(sim_payload, np.ndarray):
        return (
            sim_payload.ndim == 2
            and int(sim_payload.shape[0]) == int(n_selected)
            and int(sim_payload.shape[1]) == int(n_selected)
        )

    if isinstance(sim_payload, dict) and sim_payload.get('format') == 'svd_gram':
        z = sim_payload.get('z', None)
        if z is None:
            return False
        z_arr = np.asarray(z)
        shp = _similarity_shape(sim_payload)
        return z_arr.ndim == 2 and int(z_arr.shape[0]) == int(n_selected) and shp == (int(n_selected), int(n_selected))

    return False


def build_global_subgraph_pool_records(datasets, target_graph_ids_zero_based):
    """Build merged subgraph records across datasets and run stage2-style selection.

    Returns:
        selected_samples: selected subgraph sample metadata
        sample_similarity: cosine similarity over selected samples' structural feature vectors
    """
    cache_path_run = os.path.join(context_outputs_root, 'global_subgraph_pool.pt')
    cache_dir_global = os.path.join(parent_directory, 'cache', 'global_pool')
    os.makedirs(cache_dir_global, exist_ok=True)
    cache_path_global = os.path.join(cache_dir_global, 'global_subgraph_pool.pt')
    run_stage1_dir = os.path.join(stage1_outputs_root, 'global_merged')
    run_stage1_selected_dir = os.path.join(stage1_outputs_root, 'global_merged_selected')
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
        'stage1_cluster_method': str(args.stage1_cluster_method),
        'stage1_encoding_mode': str(args.stage1_encoding_mode),
        'stage1_use_log1p': int(args.stage1_use_log1p),
        'stage1_remove_selfloops': int(args.stage1_remove_selfloops),
        'stage2_top_m_per_cluster': int(args.stage2_top_m_per_cluster),
        'stage2_alpha': float(args.stage2_alpha),
        'stage2_beta': float(args.stage2_beta),
        'stage2_gamma': float(args.stage2_gamma),
        'stage2_top_ratio': float(args.stage2_top_ratio),
        'stage2_allow_duplicates': 1,
        'mask_source': 'sample_similarity',
        'seed': int(args.seed),
    }

    debug_trace(
        'global_pool.start',
        datasets=len(datasets),
        target_graph_ids=target_graph_ids_zero_based,
        force_rebuild_global_pool=int(args.force_rebuild_global_pool),
        stage1_num_clusters=int(args.stage1_num_clusters),
        stage1_cluster_method=str(args.stage1_cluster_method),
        stage2_top_ratio=float(args.stage2_top_ratio),
    )

    if args.force_rebuild_global_pool != 1:
        debug_trace('global_pool.cache_lookup.start', cache_paths=2)
        for cache_path in [cache_path_global, cache_path_run]:
            if not os.path.exists(cache_path):
                continue
            try:
                cached = torch_load_trusted(cache_path)
                if isinstance(cached, dict) and cached.get('signature', None) == cache_signature:
                    selected_samples = cached.get('selected_samples', [])
                    sample_similarity = cached.get('sample_similarity', None)
                    if _is_valid_similarity_payload(sample_similarity, len(selected_samples)):
                        print(
                            f'[global_pool] loaded cache: {cache_path} '
                            f'(selected={len(selected_samples)})'
                        )
                        ds_cov = len({str(s['dataset_name']) for s in selected_samples})
                        cl_cov = len({int(s['cluster_id']) for s in selected_samples})
                        debug_trace(
                            'global_pool.cache_hit',
                            selected=len(selected_samples),
                            dataset_coverage=ds_cov,
                            cluster_coverage=cl_cov,
                            similarity_shape=_similarity_shape(sample_similarity),
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

                            selected_eval = export_stage1_selected_pca_same_axes(
                                stage1_dir=run_stage1_dir,
                                selected_sample_ids=[int(s['sample_id']) for s in selected_samples],
                                output_dir=run_stage1_selected_dir,
                            )
                            print(f"[stage1-metrics] global_merged_selected: {selected_eval['selected_pca_png']}")
                        except Exception as e:
                            print(f"[stage1-metrics] warn global_merged(cache): {e}")

                        try:
                            stage2_eval = evaluate_stage2_artifacts(run_stage2_dir)
                            print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")
                        except Exception as e:
                            print(f"[stage2-metrics] warn global_merged(cache): {e}")

                        debug_trace('global_pool.cache_hit.done')
                        return selected_samples, sample_similarity
            except Exception as e:
                print(f'[global_pool] warn: failed to load cache {cache_path}: {e}')
        debug_trace('global_pool.cache_lookup.miss')

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

            before_cnt = len(merged_records)
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
            debug_trace(
                'global_pool.graph.done',
                dataset=ds,
                graph_id=int(graph_id),
                subgraphs=len(subgraphs),
                newly_added=(len(merged_records) - before_cnt),
                total_samples=len(merged_records),
            )
        debug_trace(
            'global_pool.dataset.done',
            dataset=ds,
            dataset_total_subgraphs=dataset_subgraph_counts[ds],
        )

    if len(merged_records) == 0:
        raise RuntimeError('Global subgraph pool is empty. Please check dataset/graphId settings.')

    print(
        f"[debug] subgraph build done: total_subgraphs={len(merged_records)}, "
        f"per_dataset={dataset_subgraph_counts}"
    )

    debug_trace('global_pool.clustering.start', total_subgraphs=len(merged_records))
    labels, centers_raw, _, _, _, _ = fit_clusters(
        records=merged_records,
        num_clusters=args.stage1_num_clusters,
        seed=args.seed,
        use_log1p=bool(args.stage1_use_log1p),
        encoding_mode=args.stage1_encoding_mode,
        cluster_method=args.stage1_cluster_method,
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

    debug_trace('global_pool.representative_pool.start', feature_dims=len(FEATURE_KEYS))
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
    debug_trace(
        'global_pool.representative_pool.done',
        representative_pool=int(rep_indices.size),
        learnability_ranked=int(len(order)),
    )

    # Stage-2: second selection must come from remaining pool only.
    all_idx = np.arange(len(merged_records), dtype=np.int64)
    rep_indices_stage1 = np.unique(rep_indices.astype(np.int64)) if rep_indices.size > 0 else np.array([], dtype=np.int64)
    stage1_set = set(int(i) for i in rep_indices_stage1.tolist())
    remain_idx = np.array([i for i in all_idx.tolist() if int(i) not in stage1_set], dtype=np.int64)

    top_ratio = max(0.0, min(1.0, float(args.stage2_top_ratio)))
    second_take = int(np.ceil(remain_idx.size * top_ratio)) if remain_idx.size > 0 else 0
    if remain_idx.size > 0 and second_take > 0:
        remain_order = remain_idx[np.argsort(-learnability[remain_idx])]
        rep_indices_stage2 = remain_order[:second_take]
    else:
        rep_indices_stage2 = np.array([], dtype=np.int64)

    selected_idx = np.concatenate([rep_indices_stage1, rep_indices_stage2], axis=0)
    if selected_idx.size > 0:
        selected_idx = np.unique(selected_idx)

    def _cluster_ratio(indices):
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.size == 0:
            return {}
        ids = cluster_ids[idx_arr]
        uniq, cnt = np.unique(ids, return_counts=True)
        total = float(max(1, cnt.sum()))
        return {str(int(c)): float(n / total) for c, n in zip(uniq.tolist(), cnt.tolist())}

    def _cluster_count(indices):
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.size == 0:
            return {}
        ids = cluster_ids[idx_arr]
        uniq, cnt = np.unique(ids, return_counts=True)
        return {str(int(c)): int(n) for c, n in zip(uniq.tolist(), cnt.tolist())}

    stage_debug = {
        'num_samples_total': int(len(merged_records)),
        'stage1_top_m_per_cluster': int(args.stage2_top_m_per_cluster),
        'stage2_top_p_percent': float(top_ratio),
        'stage1_selected_count': int(rep_indices_stage1.size),
        'stage2_selected_count': int(rep_indices_stage2.size),
        'selected_total_count': int(selected_idx.size),
        'stage1_cluster_count': _cluster_count(rep_indices_stage1),
        'stage2_cluster_count': _cluster_count(rep_indices_stage2),
        'selected_total_cluster_count': _cluster_count(selected_idx),
        'stage1_cluster_ratio': _cluster_ratio(rep_indices_stage1),
        'stage2_cluster_ratio': _cluster_ratio(rep_indices_stage2),
        'selected_total_cluster_ratio': _cluster_ratio(selected_idx),
    }

    print(
        f"[debug] sample pool done: stage1_selected={stage_debug['stage1_selected_count']}, "
        f"stage2_selected={stage_debug['stage2_selected_count']}, "
        f"selected_total={stage_debug['selected_total_count']}, "
        f"top_ratio={top_ratio:.4f}"
    )
    rank_cluster_unique = len(stage_debug['stage2_cluster_count'])
    print(
        f"[debug] rank cluster diversity: rank_selected={stage_debug['stage2_selected_count']}, "
        f"unique_clusters={rank_cluster_unique}"
    )
    print(f"[debug] stage1_cluster_count={stage_debug['stage1_cluster_count']}")
    print(f"[debug] stage2_cluster_count={stage_debug['stage2_cluster_count']}")
    print(f"[debug] selected_total_cluster_count={stage_debug['selected_total_cluster_count']}")
    print(f"[debug] stage1_cluster_ratio={stage_debug['stage1_cluster_ratio']}")
    print(f"[debug] stage2_cluster_ratio={stage_debug['stage2_cluster_ratio']}")
    print(f"[debug] selected_total_cluster_ratio={stage_debug['selected_total_cluster_ratio']}")

    rep_index_set = set(int(i) for i in rep_indices_stage1.tolist())

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
    for idx in selected_idx.tolist():
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

    selected_idx_arr = np.asarray(selected_idx, dtype=np.int64)
    selected_x = x[selected_idx_arr]
    selected_norm = np.linalg.norm(selected_x, axis=1, keepdims=True)
    selected_norm = np.where(selected_norm < 1e-12, 1.0, selected_norm)
    selected_unit = selected_x / selected_norm
    # Use low-rank SVD payload instead of dense N x N matrix to avoid OOM.
    sample_similarity = _build_svd_similarity_payload(
        selected_unit,
        max_rank=max(1, int(args.global_pool_svd_rank)),
    )
    if isinstance(sample_similarity, dict):
        debug_trace(
            'global_pool.svd_payload',
            rank=int(sample_similarity.get('rank', -1)),
            shape=tuple(sample_similarity.get('shape', (0, 0))),
        )

    # Persist global stage2 diagnostics/visualization (before vs after selection).
    global_stage2_dir = run_stage2_dir
    os.makedirs(global_stage2_dir, exist_ok=True)
    in_pool = np.zeros(len(merged_records), dtype=np.int64)
    if selected_idx.size > 0:
        in_pool[selected_idx] = 1

    rep_dist_stage1 = np.full((rep_indices_stage1.size,), -1.0, dtype=np.float64)
    if rep_indices_stage1.size > 0:
        centers_used = centers_raw[cluster_ids[rep_indices_stage1]]
        rep_dist_stage1 = np.linalg.norm(x[rep_indices_stage1] - centers_used, axis=1)
    rep_dist = np.concatenate(
        [
            rep_dist_stage1,
            np.full((rep_indices_stage2.size,), -1.0, dtype=np.float64),
        ],
        axis=0,
    )

    rep_indices_out = np.concatenate([rep_indices_stage1, rep_indices_stage2], axis=0)
    decoded_summary = {
        'enabled': False,
        'decoded_count': 0,
        'decoded_root': '',
        'dataset_prompt_file': '',
        'reason': 'global_merged includes multiple datasets; decode is handled in per-dataset stage2.',
    }

    stage2_write = write_stage2_outputs(
        output_dir=Path(global_stage2_dir),
        input_csv=Path(global_stage1_csv),
        sample_ids=np.asarray([int(r['sample_id']) for r in merged_records], dtype=np.int64),
        graph_ids=np.asarray([int(r['graph_id']) for r in merged_records], dtype=np.int64),
        subgraph_ids=np.asarray([int(r['subgraph_id']) for r in merged_records], dtype=np.int64),
        cluster_ids=cluster_ids,
        rep_indices=rep_indices_out,
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
        top_m_per_cluster=int(args.stage2_top_m_per_cluster),
        top_p_percent=float(top_ratio),
        alpha=float(args.stage2_alpha),
        beta=float(args.stage2_beta),
        gamma=float(args.stage2_gamma),
        stage_debug=stage_debug,
        decoded_summary=decoded_summary,
    )
    global_stage2_csv = stage2_write['scores_csv']

    try:
        stage1_eval = evaluate_stage1_artifacts(global_stage1_dir)
        print(f"[stage1-metrics] global_merged: {stage1_eval['metrics_json']}")

        selected_eval = export_stage1_selected_pca_same_axes(
            stage1_dir=global_stage1_dir,
            selected_sample_ids=[int(s['sample_id']) for s in selected_samples],
            output_dir=run_stage1_selected_dir,
        )
        print(f"[stage1-metrics] global_merged_selected: {selected_eval['selected_pca_png']}")
    except Exception as e:
        print(f"[stage1-metrics] warn global_merged: {e}")

    try:
        stage2_eval = evaluate_stage2_artifacts(global_stage2_dir)
        print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")
    except Exception as e:
        print(f"[stage2-metrics] warn global_merged: {e}")

    print(
        f'[global_pool] total_subgraphs={len(merged_records)}, '
        f'stage1_pool={int(rep_indices_stage1.size)}, '
        f'stage2_pool={int(rep_indices_stage2.size)}, '
        f'selected={len(selected_samples)}, '
        f'top_ratio={top_ratio:.4f}'
    )
    debug_trace(
        'global_pool.select.done',
        total_subgraphs=len(merged_records),
        representative_pool=int(selected_idx.size),
        selected=len(selected_samples),
    )

    cache_payload = {
        'signature': cache_signature,
        'selected_samples': selected_samples,
        'sample_similarity': sample_similarity,
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
            torch.save(cache_payload, cache_path, pickle_protocol=4)
            print(f'[global_pool] saved cache: {cache_path}')
        except Exception as e:
            print(f'[global_pool] warn: failed to save cache {cache_path}: {e}')

    debug_trace('global_pool.done')
    return selected_samples, sample_similarity


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
    egraphcl_temperature=args.egraphcl_temperature,
    egraphcl_pair_chunk_size=args.egraphcl_pair_chunk_size,
    egraphcl_candidate_pool=args.egraphcl_candidate_pool,
    egraphcl_soft_topk=args.egraphcl_soft_topk,
    egraphcl_max_negatives=args.egraphcl_max_negatives,
    egraphcl_mask_source=args.egraphcl_mask_source,
    egraphcl_online_tau=args.egraphcl_tau,
    egraphcl_batch_size=args.egraphcl_batch_size,
    egraphcl_min_negatives_per_anchor=args.egraphcl_min_negatives_per_anchor,
    egraphcl_structure_min_negatives_per_anchor=args.egraphcl_structure_min_negatives_per_anchor,
    egraphcl_max_mask_ratio=args.egraphcl_max_mask_ratio,
    egraphcl_curve_bins=args.egraphcl_curve_bins,
    egraphcl_curve_chunk_rows=args.egraphcl_curve_chunk_rows,
).cuda()

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
sample_dataset_ids = None
selected_samples_for_flow = []
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))
except:
    # ------------------- 预训练数据准备（特征/邻接/增强/负采样）-------------------
    debug_trace(
        'pretrain.prepare.start',
        pretrain_method=args.pretrain_method,
        global_subgraph_pool=int(args.global_subgraph_pool),
        target_graph_id=target_graph_id,
    )
    if args.global_subgraph_pool == 1 and args.pretrain_method in ('GRAPHCL', 'EGRAPHCL'):
        target_graph_id_zero_based = [int(gid) - 1 for gid in target_graph_id]
        selected_samples, global_sample_similarity = build_global_subgraph_pool_records(
            pretrain_dataset_names,
            target_graph_id_zero_based,
        )
        debug_trace(
            'pretrain.prepare.global_pool.loaded',
            selected_samples=len(selected_samples),
            similarity_shape=_similarity_shape(global_sample_similarity),
        )
        selected_samples_for_flow = selected_samples

        if args.pretrain_method == 'EGRAPHCL':
            print(
                '[egraphcl] online pair mode enabled: '
                f'tau={float(args.egraphcl_tau):.4f}, '
                'pair mask is built from prompt-conditioned sample embeddings in model forward (fixed tau; no scheduling).'
            )

        full_graph_cache = {}
        sample_dataset_ids = []
        for sample_idx, sample in enumerate(selected_samples, start=1):
            ds_name = sample['dataset_name']
            ds_idx = sample['dataset_idx']
            graph_id = sample['graph_id']
            node_idx = sample['node_idx']
            cluster_id = sample['cluster_id']

            if _should_log_progress(sample_idx, len(selected_samples), args.sample_progress_every):
                debug_trace(
                    'pretrain.prepare.sample_progress',
                    current=sample_idx,
                    total=len(selected_samples),
                    dataset=ds_name,
                    dataset_idx=int(ds_idx),
                    graph_id=int(graph_id),
                    cluster_id=int(cluster_id),
                    nodes=int(len(node_idx)),
                )

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
                # Strict sample-level EGRAPHCL path: node-level soft masking is removed.
                aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(adj, feature, sparse, drop_percent)

            aug_features.append(aug_feature)
            aug_adjs.append(aug_adj)
            lbls.append(lbl)
            bce_masks.append(bce_mask)
            if args.pretrain_method == 'EGRAPHCL':
                masked_pairs, total_pairs = _count_soft_mask_pairs(bce_mask)
                egraphcl_soft_masked_pairs += masked_pairs
                egraphcl_negative_pairs += total_pairs
            features.append(feature)
            adjs.append(adj)
            sample_dataset_ids.append(int(ds_idx))

        debug_trace(
            'pretrain.prepare.global_pool.done',
            features=len(features),
            adjs=len(adjs),
            aug_features=len(aug_features),
            lbls=len(lbls),
            bce_masks=len(bce_masks),
        )

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
                    if args.pretrain_method == 'EGRAPHCL':
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
        if len(aug_features) <= 1:
            raise RuntimeError('EGRAPHCL online pair mode requires at least 2 samples.')
        print(
            '[egraphcl] strict sample-level online mode: '
            f'samples={len(aug_features)}, tau={float(args.egraphcl_tau):.4f}'
        )

    debug_trace(
        'flow_check.start',
        aug_features=len(aug_features),
        selected_samples=len(selected_samples_for_flow),
    )
    flow_ok, flow_report_path = run_flow_sanity_checks(
        args=args,
        result_dir=result_dir,
        stage1_outputs_root=stage1_outputs_root,
        stage2_outputs_root=stage2_outputs_root,
        aug_features=aug_features,
        aug_adjs=aug_adjs,
        lbls=lbls,
        bce_masks=bce_masks,
        sample_dataset_ids=sample_dataset_ids,
        selected_samples=selected_samples_for_flow,
    )
    debug_trace('flow_check.done', ok=bool(flow_ok), report=flow_report_path)
    if int(args.flow_check_only) == 1:
        if not flow_ok:
            raise RuntimeError(f'Flow sanity check failed. See report: {flow_report_path}')
        print('[flow-check] flow_check_only=1, all checks passed. Exit before training loop.')
        raise SystemExit(0)

    # ------------------- 优化器 & 搬到 GPU -------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',
        factor=float(args.lr_scheduler_factor),
        patience=int(args.lr_scheduler_patience),
        min_lr=float(args.lr_scheduler_min_lr),
    )
    prev_lr = float(optimiser.param_groups[0]['lr'])
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

        # EGRAPHCL 保持样本池在 CPU，按小批次在 model.forward 内逐步搬运到 GPU。
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            features = [tensors.cuda() for tensors in features]
            adjs = [
                process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
                for adj in adjs
            ]
            negetive_samples = [tensors.cuda() for tensors in negetive_samples]

            # LP 情况下 negetive_samples 可能为空，此时使用合并图上的 negetive_sample
            if len(negetive_samples) == 0:
                negetive_samples = negetive_sample.cuda()

        elif args.pretrain_method == 'GRAPHCL':
            lbls = [tensors.cuda() for tensors in lbls]
            bce_masks = [tensors.cuda() for tensors in bce_masks]
            aug_adjs = [tensors.cuda() for tensors in aug_adjs]
            aug_features = [tensors.cuda() for tensors in aug_features]

    debug_trace(
        'train_loop.start',
        nb_epochs=int(nb_epochs),
        patience=int(patience),
        lr=float(lr),
        pretrain_method=args.pretrain_method,
    )

    print('Start training...')
    curve_dir = os.path.join(stage2_outputs_root, 'global_merged', 'egraphcl_similarity_curves')
    log_cuda_peak_mem = bool(int(args.log_cuda_peak_mem)) and torch.cuda.is_available()
    if log_cuda_peak_mem:
        torch.cuda.reset_peak_memory_stats()

    # ------------------- 预训练循环（early stopping）-------------------
    for epoch in range(nb_epochs):

        loss = 0
        model.train()
        optimiser.zero_grad()

        # GRAPHCL / EGRAPHCL: 传入 (aug_features, aug_adjs, lbls, bce_masks)
        if args.pretrain_method == 'GRAPHCL' or args.pretrain_method == 'EGRAPHCL':
            if args.pretrain_method == 'EGRAPHCL':
                total_samples = int(len(aug_features))
                batch_graphs = max(2, int(args.egraphcl_batch_size))
                order = list(torch.randperm(total_samples).tolist())
                num_effective_batches = sum(
                    1
                    for start in range(0, total_samples, batch_graphs)
                    if (min(start + batch_graphs, total_samples) - start) > 1
                )
                if num_effective_batches <= 0:
                    raise RuntimeError('EGRAPHCL requires at least 2 samples per optimization epoch.')

                should_dump_curve = (
                    int(args.egraphcl_curve_log_every) > 0
                    and ((epoch + 1) % int(args.egraphcl_curve_log_every) == 0)
                )

                loss_weighted_sum = 0.0
                total_valid_anchors = 0
                agg_masked_pairs = 0
                agg_total_pairs = 0
                agg_active_negative_pairs = 0
                curve_dumped = False
                non_finite = False

                for start in range(0, total_samples, batch_graphs):
                    end = min(start + batch_graphs, total_samples)
                    batch_indices = order[start:end]
                    if len(batch_indices) <= 1:
                        continue

                    batch_aug_features = [aug_features[idx] for idx in batch_indices]
                    batch_aug_adjs = [aug_adjs[idx] for idx in batch_indices]
                    batch_lbls = [lbls[idx] for idx in batch_indices]
                    batch_bce_masks = [bce_masks[idx] for idx in batch_indices]
                    batch_sample_dataset_ids = None
                    if sample_dataset_ids is not None:
                        batch_sample_dataset_ids = [sample_dataset_ids[idx] for idx in batch_indices]

                    model.egraphcl_collect_curve = bool(should_dump_curve and (not curve_dumped))
                    batch_loss = model(
                        batch_aug_features,
                        batch_aug_adjs,
                        sparse,
                        None,
                        None,
                        None,
                        batch_lbls,
                        batch_bce_masks,
                        None,
                        pretrain_method=args.pretrain_method,
                        sample_dataset_ids=batch_sample_dataset_ids,
                    )

                    if not torch.isfinite(batch_loss):
                        print(f'[train] non-finite egraphcl batch loss at epoch {epoch + 1}: {batch_loss}')
                        non_finite = True
                        break

                    # 以 batch 数做梯度归一，避免 batch 切分改变有效学习率。
                    (batch_loss / float(num_effective_batches)).backward()

                    stats = getattr(model, 'last_egraphcl_stats', None)
                    if isinstance(stats, dict):
                        valid_anchors = int(stats.get('valid_anchors', len(batch_indices)))
                        if valid_anchors > 0:
                            loss_weighted_sum += float(batch_loss.detach().item()) * float(valid_anchors)
                            total_valid_anchors += valid_anchors
                        agg_masked_pairs += int(stats.get('masked_pairs', 0))
                        agg_total_pairs += int(stats.get('total_pairs', 0))
                        agg_active_negative_pairs += int(stats.get('active_negative_pairs', 0))

                    if model.egraphcl_collect_curve:
                        curve_stats = getattr(model, 'last_egraphcl_similarity_curve', None)
                        outputs = dump_similarity_curves(
                            curve_stats,
                            curve_dir,
                            epoch + 1,
                            save_plot=bool(int(args.egraphcl_curve_save_plot)),
                        )
                        if outputs:
                            print(f"[egraphcl-curve] epoch={epoch + 1} saved_files={len(outputs)}")
                        curve_dumped = True
                        model.egraphcl_collect_curve = False

                if non_finite:
                    break

                if total_valid_anchors > 0:
                    loss_value_egraphcl = loss_weighted_sum / float(total_valid_anchors)
                else:
                    loss_value_egraphcl = 0.0
                loss = torch.tensor(loss_value_egraphcl, device=next(model.parameters()).device)

                egraphcl_log_epoch = (
                    (epoch + 1) <= 3
                    or (
                        int(args.egraphcl_mask_log_every) > 0
                        and ((epoch + 1) % int(args.egraphcl_mask_log_every) == 0)
                    )
                )
                if egraphcl_log_epoch:
                    mask_ratio = float(agg_masked_pairs / max(1, agg_total_pairs))
                    print(
                        '[egraphcl-mask] '
                        'branch=single_with_dual_prompt '
                        'source=encoded_similarity '
                        f'batch={batch_graphs} '
                        f'mask_ratio={mask_ratio:.4f} '
                        f'masked={int(agg_masked_pairs)}/{int(agg_total_pairs)} '
                        f'active_neg={int(agg_active_negative_pairs)}'
                    )
            else:
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

        if not torch.isfinite(loss):
            print(f'[train] non-finite loss detected at epoch {epoch + 1}: {loss}')
            break

        if args.pretrain_method != 'EGRAPHCL':
            loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
        optimiser.step()

        loss_value = float(loss.detach().item())
        epoch_log = (
            (epoch + 1) <= 5
            or (
                int(args.epoch_log_every) > 0
                and ((epoch + 1) % int(args.epoch_log_every) == 0)
            )
        )
        if epoch_log:
            print('Loss:[{:.8f}]'.format(loss_value))
            if log_cuda_peak_mem:
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                print(
                    f'[mem] epoch={epoch + 1} '
                    f'peak_alloc={peak_alloc_gb:.2f}GB '
                    f'peak_reserved={peak_reserved_gb:.2f}GB'
                )
                torch.cuda.reset_peak_memory_stats()
        lr_scheduler.step(loss_value)
        current_lr = float(optimiser.param_groups[0]['lr'])
        if current_lr < prev_lr:
            print(f'[train] lr reduced: {prev_lr:.8g} -> {current_lr:.8g} (epoch={epoch + 1})')
            prev_lr = current_lr

        # 基于 loss 的 early stopping，并保存最优 checkpoint
        if loss_value < (best - float(args.early_stop_min_delta)):
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
        if epoch_log:
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