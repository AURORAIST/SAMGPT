from pathlib import Path
import sys
import warnings
import argparse
import csv
import logging
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# -----------------------------
# 路径初始化
# -----------------------------
# 当前脚本所在目录
CURRENT_DIR = Path(__file__).resolve().parent
# 当前目录的上一级目录，通常作为源码根目录之一
SRC_DIR = CURRENT_DIR.parent

# 将源码目录加入 sys.path，保证本地模块可导入
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprompt import PrePrompt, pca_compression
from utils.dataset import load_dataset
from utils import process
from utils import aug
from tools.stage_metrics import evaluate_stage1_artifacts, evaluate_stage2_artifacts
from models.stage1_feature_schema import DEFAULT_RAW9_FEATURE_MASK, normalize_raw9_feature_mask
from train.context_stage_artifacts import ContextArtifactsManager
from train.context_filter_stage2 import Stage2GraphFilter
from train.context_filter_struct import StructContextMapper
from train.context_global_pool import GlobalSubgraphPoolBuilder, GlobalPrototypeMapper, pca_compression_subgraph
from train.stage_pipeline import encode_subgraph, get_stage1_reference_graph, split_graph_to_subgraphs
from train.struct_prompt_router import StructPromptRouter, StructPromptRouterConfig

# 忽略大部分 warning，避免训练日志被无关提示刷屏
warnings.filterwarnings("ignore")

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

# -------------------------------------------------------------------
# 为了让 nohup / 重定向日志时输出更及时，这里开启 stdout/stderr 行缓冲
# 某些环境下默认是块缓冲，训练过程中的 print 可能要过很久才刷出来
# -------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# -------------------------------------------------------------------
# PyTorch 2.6+ compatibility
# 从 PyTorch 2.6 开始，torch.load 默认参数 weights_only=True，
# 这会阻止某些非 Tensor 的对象反序列化。
#
# 但本项目缓存文件中可能包含 scipy.sparse.csr_matrix 等对象，
# 因此对于“本地可信文件”，显式使用 weights_only=False。
# -------------------------------------------------------------------


def torch_load_trusted(path: str, map_location=None):
    """加载本地可信缓存/检查点，允许 pickle 任意对象。"""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_trusted(path: str, map_location=None):
    """加载 state_dict，并做基本类型检查。"""
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd


# 先清一次 GPU cache，减少历史残留占用
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
parser.add_argument('--pretrain_method', type=str, default="EGRAPHCL", help='EGRAPHCL')

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
parser.add_argument('--stage1_feature_k', type=int, default=0, help='raw9 ablation: keep first k features (3..9); 0 means use all raw9 dims')
parser.add_argument('--stage1_feature_mask', type=str, default='123456789', help='raw9 feature mask by digit index (e.g., 123456789, 13579, 1,3,5); only works with raw9')
parser.add_argument('--stage1_use_log1p', type=int, default=1, help='1: log1p skewed count features before scaling in stage1')
parser.add_argument('--stage1_remove_selfloops', type=int, default=1, help='1: remove self-loops in stage1; 0: keep self-loops')

# Stage2 自动构建参数
parser.add_argument('--stage2_top_m_per_cluster', type=int, default=5, help='representative pool size per cluster for stage2 auto-prepare')
parser.add_argument('--stage2_alpha', type=float, default=1.0, help='stage2 novelty weight')
parser.add_argument('--stage2_beta', type=float, default=1.0, help='stage2 coverage weight')
parser.add_argument('--stage2_gamma', type=float, default=1.0, help='stage2 redundancy weight')

# EGRAPHCL 专用参数
parser.add_argument('--egraphcl_tau', type=float, default=0.85, help='soft-positive threshold on structural cosine similarity for EGRAPHCL')
parser.add_argument('--global_subgraph_pool', type=int, default=1, help='1: unified cross-dataset subgraph pool for EGRAPHCL')
parser.add_argument('--force_rebuild_global_pool', type=int, default=0, help='1: rebuild merged global subgraph pool cache')
parser.add_argument('--pretrain_batch_size', type=int, default=64, help='mini-batch size for EGRAPHCL pretraining')
parser.add_argument('--rebuild_run_cache', type=int, default=1, help='1: rebuild per-run cache under current result dir to avoid stale cache')
parser.add_argument('--pretrain_max_epochs', type=int, default=100, help='max pretraining epochs for EGRAPHCL')
parser.add_argument('--pretrain_patience', type=int, default=20, help='early stopping patience for pretraining')
parser.add_argument('--pretrain_min_delta', type=float, default=1e-4, help='minimum loss improvement to reset pretraining early stop counter')
parser.add_argument('--egraphcl_negative_pair_mode', type=str, default='topk', choices=['full', 'topk', 'random'], help='negative pair usage mode in EGRAPHCL contrastive loss')
parser.add_argument('--egraphcl_negative_topk', type=int, default=16, help='top-k negatives per anchor when egraphcl_negative_pair_mode=topk')
parser.add_argument('--egraphcl_negative_random_ratio', type=float, default=0.25, help='negative sampling ratio per anchor when egraphcl_negative_pair_mode=random')
parser.add_argument('--egraphcl_mask_strategy', type=str, default='quantile', choices=['fixed', 'quantile'], help='strategy used to build EGRAPHCL sample-pair mask when no external mask is provided')
parser.add_argument('--egraphcl_mask_quantile', type=float, default=0.9, help='quantile used to derive adaptive mask threshold when egraphcl_mask_strategy=quantile')
parser.add_argument('--egraphcl_mask_min_tau', type=float, default=0.6, help='lower clamp for adaptive EGRAPHCL mask threshold')
parser.add_argument('--egraphcl_mask_max_tau', type=float, default=0.98, help='upper clamp for adaptive EGRAPHCL mask threshold')
parser.add_argument('--egraphcl_min_negatives_per_row', type=int, default=2, help='minimum negatives kept for each anchor in EGRAPHCL loss')
parser.add_argument('--egraphcl_log_every', type=int, default=20, help='print EGRAPHCL batch diagnostics every N batches')
parser.add_argument('--prompt_routing_mode', type=str, default='dataset', choices=['dataset', 'prototype'], help='prompt routing mode for pretraining')
parser.add_argument('--prompt_id_split_base', type=int, default=10000, help='split base for unified route ids: dataset in [0, base), prototype in [base, +inf)')
parser.add_argument('--downstream_prompt_bank_mode', type=str, default='auto', choices=['auto', 'dataset', 'prototype'], help='prompt bank exported to downstream head')
parser.add_argument('--egraphcl_encoder_mode', type=str, default='batched', choices=['batched', 'sample'], help='encoder mode for EGRAPHCL graph-pair vectors')
parser.add_argument('--egraphcl_batched_chunk_graphs', type=int, default=32, help='micro-chunk graph count for batched encoder; 0 means no extra chunking')
parser.add_argument('--prototype_soft_prompt', type=int, default=0, help='1: enable prototype-weight soft prompt mixing; 0: use hard prototype routing (0413-compatible)')
parser.add_argument('--prototype_assign_tau', type=float, default=0.2, help='temperature used for statistical prototype soft assignment')
parser.add_argument('--prototype_consistency_loss_weight', type=float, default=0.05, help='weight of prototype consistency regularization in prototype routing')
parser.add_argument('--prototype_balance_loss_weight', type=float, default=0.01, help='weight of prototype usage balance regularization in prototype routing')
parser.add_argument('--pretrain_preprocess_cache', type=int, default=1, help='cache EGRAPHCL preprocessed subgraph tensors under run cache directory')

# 新增参数（任务7）
parser.add_argument('--node_loss_weight', type=float, default=0.5, help='weight of node-level contrastive loss')
parser.add_argument('--egraphcl_temperature_schedule', type=str, default='fixed', choices=['fixed', 'cosine'], help='temperature schedule for EGRAPHCL contrastive loss')
parser.add_argument('--egraphcl_temperature_max', type=float, default=0.5, help='max temperature for cosine annealing schedule')
parser.add_argument('--egraphcl_temperature_min', type=float, default=0.05, help='min temperature for cosine annealing schedule')
parser.add_argument('--multi_scale', type=str, default='none', choices=['none', 'sum', 'concat'], help='multi-scale GCN output aggregation mode')
parser.add_argument('--aug_type_pretrain', type=str, default='edge', choices=['edge', 'mask', 'mixed'], help='augmentation type for pretraining')

# 真正解析参数
args = parser.parse_args()

# ------------------- 参数合法性检查 -------------------
args.max_subgraph_nodes = int(args.max_subgraph_nodes)
if args.max_subgraph_nodes <= 0:
    raise ValueError('max_subgraph_nodes must be a positive integer')

if int(args.pretrain_batch_size) <= 0:
    raise ValueError('pretrain_batch_size must be a positive integer')

if int(args.pretrain_max_epochs) <= 0:
    raise ValueError('pretrain_max_epochs must be a positive integer')

if int(args.pretrain_patience) <= 0:
    raise ValueError('pretrain_patience must be a positive integer')

if float(args.pretrain_min_delta) < 0:
    raise ValueError('pretrain_min_delta must be >= 0')

if int(args.egraphcl_negative_topk) <= 0:
    raise ValueError('egraphcl_negative_topk must be a positive integer')

if float(args.egraphcl_negative_random_ratio) <= 0 or float(args.egraphcl_negative_random_ratio) > 1:
    raise ValueError('egraphcl_negative_random_ratio must be in (0, 1]')

if float(args.egraphcl_mask_quantile) < 0 or float(args.egraphcl_mask_quantile) > 1:
    raise ValueError('egraphcl_mask_quantile must be in [0, 1]')

if float(args.egraphcl_mask_min_tau) < 0 or float(args.egraphcl_mask_min_tau) > 1:
    raise ValueError('egraphcl_mask_min_tau must be in [0, 1]')

if float(args.egraphcl_mask_max_tau) < 0 or float(args.egraphcl_mask_max_tau) > 1:
    raise ValueError('egraphcl_mask_max_tau must be in [0, 1]')

if float(args.egraphcl_mask_min_tau) > float(args.egraphcl_mask_max_tau):
    raise ValueError('egraphcl_mask_min_tau must be <= egraphcl_mask_max_tau')

if int(args.egraphcl_min_negatives_per_row) <= 0:
    raise ValueError('egraphcl_min_negatives_per_row must be a positive integer')

if int(args.egraphcl_log_every) <= 0:
    raise ValueError('egraphcl_log_every must be a positive integer')

if int(args.egraphcl_batched_chunk_graphs) < 0:
    raise ValueError('egraphcl_batched_chunk_graphs must be >= 0')

if int(args.prototype_soft_prompt) not in (0, 1):
    raise ValueError('prototype_soft_prompt must be 0 or 1')

if float(args.prototype_assign_tau) <= 0:
    raise ValueError('prototype_assign_tau must be > 0')

if float(args.prototype_consistency_loss_weight) < 0:
    raise ValueError('prototype_consistency_loss_weight must be >= 0')

if float(args.prototype_balance_loss_weight) < 0:
    raise ValueError('prototype_balance_loss_weight must be >= 0')

if int(args.prompt_id_split_base) <= 0:
    raise ValueError('prompt_id_split_base must be a positive integer')

if int(args.stage1_feature_k) < 0:
    raise ValueError('stage1_feature_k must be >= 0')

if args.stage1_encoding_mode != 'raw9' and int(args.stage1_feature_k) > 0:
    raise ValueError('stage1_feature_k only works with stage1_encoding_mode=raw9')

stage1_feature_mask_normalized = normalize_raw9_feature_mask(args.stage1_feature_mask, allow_empty=True)
if args.stage1_encoding_mode != 'raw9' and stage1_feature_mask_normalized not in ('', '123456789'):
    raise ValueError('stage1_feature_mask only works with stage1_encoding_mode=raw9')

if args.stage1_encoding_mode == 'raw9' and int(args.stage1_feature_k) == 0 and stage1_feature_mask_normalized == '':
    raise ValueError('stage1_feature_mask must contain at least one digit in [1..9] when stage1_feature_k=0')

if args.stage1_encoding_mode == 'raw9' and int(args.stage1_feature_k) > 0 and stage1_feature_mask_normalized not in ('', '123456789'):
    raise ValueError('stage1_feature_k and custom stage1_feature_mask are mutually exclusive; set stage1_feature_k=0')

if args.stage1_encoding_mode == 'raw9' and int(args.stage1_feature_k) > 0:
    if int(args.stage1_feature_k) < 3 or int(args.stage1_feature_k) > 9:
        raise ValueError('stage1_feature_k for raw9 must be in [3, 9], or 0 to use all dims')

# 统一规范化，后续模块全部拿标准数字串（如 "123"）
args.stage1_feature_mask = stage1_feature_mask_normalized or DEFAULT_RAW9_FEATURE_MASK

if str(args.prompt_routing_mode).lower() == 'prototype' and int(args.prototype_soft_prompt) != 1:
    print('[context] prompt_routing_mode=prototype requires soft prototype prompt; force prototype_soft_prompt=1.')
    args.prototype_soft_prompt = 1

# -------------------------------------------------------------------
# 为了避免旧缓存污染当前实验，每次 run 默认强制重建 stage1/stage2/global pool
# -------------------------------------------------------------------
if int(args.rebuild_run_cache) == 1:
    args.force_rebuild_stage1 = 1
    args.force_rebuild_stage2 = 1
    args.force_rebuild_global_pool = 1

# EGRAPHCL 预训练消融仅支持 all/ft/st。
if args.pretrain_method == 'EGRAPHCL':
    _allowed_pre_ablation = {'all', 'ft', 'st'}
    _ablation_pre_norm = str(args.ablation_pre).lower()
    if _ablation_pre_norm not in _allowed_pre_ablation:
        raise ValueError(
            f"EGRAPHCL only supports ablation_pre in {_allowed_pre_ablation}, got '{args.ablation_pre}'"
        )
    args.ablation_pre = _ablation_pre_norm

args.ablation_down = str(args.ablation_down).lower()

# 当前 EGRAPHCL 路径依赖全局子图池构建样本；关闭时会导致无样本可训练。
if args.pretrain_method == 'EGRAPHCL' and int(args.global_subgraph_pool) != 1:
    raise ValueError('EGRAPHCL currently requires --global_subgraph_pool=1')

# 新增参数合法性校验（任务7）
if float(args.node_loss_weight) < 0.0 or float(args.node_loss_weight) > 1.0:
    raise ValueError('node_loss_weight must be in [0.0, 1.0]')

if float(args.egraphcl_temperature_max) <= float(args.egraphcl_temperature_min):
    raise ValueError('egraphcl_temperature_max must be > egraphcl_temperature_min')

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
drop_percent = args.drop_percent

# 拼接预训练数据集名，方便用于实验命名
pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_' + strs

# 打印当前进程号，便于排查多任务 / 后台运行问题
print(os.getpid())
print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))

# 指定 CUDA 设备排序方式
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 直接绑定到某一张 GPU
# 注意：若是多进程/多卡 launcher，通常要由外部调度器控制
torch.cuda.set_device(args.gpu)

# ------------------- 随机种子 -------------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)

# 固定 PyTorch 随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------- 路径准备 -------------------
# current_file_path: 当前脚本绝对路径
current_file_path = os.path.abspath(__file__)
# parent_directory: 项目根目录（假设此脚本在更深层目录下）
parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

# ------------------- 训练/模型超参 -------------------
nb_epochs = int(args.pretrain_max_epochs)  # 预训练最大 epoch
patience = int(args.pretrain_patience)     # early stopping 容忍次数
pretrain_min_delta = float(args.pretrain_min_delta)
lr = args.lr              # 学习率
l2_coef = 0.0             # L2 正则
hid_units = args.hid_units
sparse = True             # 是否使用稀疏邻接矩阵
LP = (args.pretrain_method == 'LP')

# 损失函数
xent = nn.CrossEntropyLoss()

# 激活函数名称
nonlinearity = 'prelu'

# ------------------- early stopping 相关变量 -------------------
best = 1e9
best_t = 0
cnt_wait = 0

num_layers_num = args.layers_num

# ------------------- 预训练数据缓存容器 -------------------
# 这些列表中每个元素一般对应一个样本图 / 子图
features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
bce_masks = []
negetive_sample = torch.tensor(0.0)

egraphcl_pair_soft_mask = None


print(pretrain_dataset_names)

# ------------------- 预训练图数量 -------------------
# 当前实现中，一个 dataset 可能对应多个 graph，
# 这里先按数据集数量计
num_pretrain_dataset_num = len(pretrain_dataset_names)

# 为每个预训练数据集创建 DataLoader
# 尽管很多 TU 数据集只有单图或很少图，这里统一成 loader 形式
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

# 压缩后的统一特征维度
unify_dim = args.unify_dim

# ------------------- 输出目录 / 日志 / cache -------------------
save_dir = os.path.join(parent_directory, 'checkpoints')

# 若未指定结果子目录，则使用当前时间戳
run_subdir = args.result_subdir.strip() if args.result_subdir else datetime.now().strftime('%Y%m%d_%H%M%S')

# 当前实验结果目录
result_dir = os.path.join(parent_directory, 'result', run_subdir)
logfile = os.path.join(result_dir, 'train.log')
cache_dir = os.path.join(result_dir, 'cache')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 若要求重建 cache，则删除旧 cache
if int(args.rebuild_run_cache) == 1 and os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)
os.makedirs(cache_dir, exist_ok=True)

print(f'[result] run_subdir={run_subdir}')
print(f'[result] result_dir={result_dir}')
print(f'[cache] cache_dir={cache_dir} rebuild={int(args.rebuild_run_cache)}')

# 上下文输出目录（stage1/stage2 结果）
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

# ------------------- 上下文构建/过滤器对象 -------------------
# 负责 stage1 / stage2 工件自动生成与管理
context_artifacts_manager = ContextArtifactsManager(
    parent_directory=parent_directory,
    stage1_outputs_root=stage1_outputs_root,
    stage2_outputs_root=stage2_outputs_root,
    max_subgraph_nodes=args.max_subgraph_nodes,
    stage1_subgraph_hop_n=args.stage1_subgraph_hop_n,
    stage1_num_clusters=args.stage1_num_clusters,
    stage1_encoding_mode=args.stage1_encoding_mode,
    stage1_feature_k=args.stage1_feature_k,
    stage1_feature_mask=args.stage1_feature_mask,
    seed=args.seed,
    stage1_remove_selfloops=args.stage1_remove_selfloops,
    stage1_use_log1p=args.stage1_use_log1p,
    force_rebuild_stage1=args.force_rebuild_stage1,
    force_rebuild_stage2=args.force_rebuild_stage2,
    stage2_top_m_per_cluster=args.stage2_top_m_per_cluster,
    stage2_alpha=args.stage2_alpha,
    stage2_beta=args.stage2_beta,
    stage2_gamma=args.stage2_gamma,
    evaluate_stage1_artifacts=evaluate_stage1_artifacts,
    evaluate_stage2_artifacts=evaluate_stage2_artifacts,
)

# 负责根据 stage2 得分对图/节点进行过滤
stage2_graph_filter = Stage2GraphFilter(
    parent_directory=parent_directory,
    stage2_outputs_root=stage2_outputs_root,
    max_subgraph_nodes=args.max_subgraph_nodes,
    stage1_subgraph_hop_n=args.stage1_subgraph_hop_n,
    stage1_remove_selfloops=args.stage1_remove_selfloops,
    stage2_top_ratio=args.stage2_top_ratio,
    torch_load_trusted=torch_load_trusted,
)

# 负责结构上下文映射，如 cluster similarity、结构标签等
struct_context_mapper = StructContextMapper(
    parent_directory=parent_directory,
    stage1_outputs_root=stage1_outputs_root,
    stage2_outputs_root=stage2_outputs_root,
    max_subgraph_nodes=args.max_subgraph_nodes,
    stage1_subgraph_hop_n=args.stage1_subgraph_hop_n,
    stage1_remove_selfloops=args.stage1_remove_selfloops,
)

# 负责构建全局子图池，用于跨数据集 EGRAPHCL
global_pool_builder = GlobalSubgraphPoolBuilder(
    parent_directory=parent_directory,
    cache_dir=cache_dir,
    stage1_outputs_root=stage1_outputs_root,
    stage2_outputs_root=stage2_outputs_root,
    max_subgraph_nodes=args.max_subgraph_nodes,
    stage1_subgraph_hop_n=args.stage1_subgraph_hop_n,
    stage1_num_clusters=args.stage1_num_clusters,
    stage1_encoding_mode=args.stage1_encoding_mode,
    stage1_feature_k=args.stage1_feature_k,
    stage1_feature_mask=args.stage1_feature_mask,
    stage1_use_log1p=args.stage1_use_log1p,
    stage1_remove_selfloops=args.stage1_remove_selfloops,
    stage2_top_m_per_cluster=args.stage2_top_m_per_cluster,
    stage2_alpha=args.stage2_alpha,
    stage2_beta=args.stage2_beta,
    stage2_gamma=args.stage2_gamma,
    stage2_top_ratio=args.stage2_top_ratio,
    prototype_assign_tau=args.prototype_assign_tau,
    seed=args.seed,
    force_rebuild_global_pool=args.force_rebuild_global_pool,
    torch_load_trusted=torch_load_trusted,
    evaluate_stage1_artifacts=evaluate_stage1_artifacts,
    evaluate_stage2_artifacts=evaluate_stage2_artifacts,
)


def ensure_context_artifacts(datasets, *, build_stage2=True):
    """确保给定数据集的上下文工件已经构建完成。"""
    return context_artifacts_manager.ensure(datasets, build_stage2=build_stage2)


def get_struct_cluster_similarity(dataset_name):
    """获取某个数据集的结构 cluster similarity。"""
    return struct_context_mapper.get_struct_cluster_similarity(dataset_name)


def maybe_filter_with_stage2_scores(dataset_name, step_id, feature, adj):
    """按 stage2 得分决定是否对当前图进行过滤。"""
    return stage2_graph_filter.maybe_filter(dataset_name, step_id, feature, adj)


def build_global_subgraph_pool_records(datasets, target_graph_ids_zero_based):
    """构建全局子图池记录，并更新全局原型中心路径（任务9）。"""
    global global_proto_centers_scaled_path, global_scaler_mean_path, global_scaler_scale_path, global_feature_mask
    result = global_pool_builder.build(datasets, target_graph_ids_zero_based)
    # 更新全局原型中心路径，供下游 get_struct_node_labels 使用
    _candidate = os.path.join(stage1_outputs_root, 'global_merged', 'prototype_centers_scaled.npy')
    _scaler_mean = os.path.join(stage1_outputs_root, 'global_merged', 'scaler_mean.npy')
    _scaler_scale = os.path.join(stage1_outputs_root, 'global_merged', 'scaler_scale.npy')
    _feature_mask_path = os.path.join(stage1_outputs_root, 'global_merged', 'feature_mask.txt')
    if os.path.exists(_candidate):
        global_proto_centers_scaled_path = _candidate
        global_scaler_mean_path = _scaler_mean
        global_scaler_scale_path = _scaler_scale
        # 读取 feature_mask
        if os.path.exists(_feature_mask_path):
            with open(_feature_mask_path, 'r') as f:
                global_feature_mask = f.read().strip()
        else:
            global_feature_mask = args.stage1_feature_mask
        print(f'[proto-align] global prototype centers (scaled) loaded from: {_candidate}')
        print(f'[proto-align] global scaler mean/scale loaded from: {_scaler_mean}, {_scaler_scale}')
        print(f'[proto-align] global feature_mask: {global_feature_mask}')
    else:
        print(f'[proto-align] warn: prototype_centers_scaled.npy not found at {_candidate}')
    return result


def get_struct_node_labels(dataset_name, data_obj, step_id):
    """获取节点的结构标签，用于增强/对比学习。"""
    return struct_context_mapper.get_struct_node_labels(
        dataset_name,
        data_obj.num_nodes,
        step_id,
        global_proto_centers_scaled_path=global_proto_centers_scaled_path,
        global_scaler_mean_path=global_scaler_mean_path,
        global_scaler_scale_path=global_scaler_scale_path,
        global_feature_mask=global_feature_mask,
        proto_tau=args.prototype_assign_tau,
    )


def get_struct_node_prototype_weights(dataset_name, data_obj, step_id):
    """实时计算下游节点到全局原型的软分配权重（不依赖 stage1 CSV）。"""
    global downstream_proto_mapper_cache

    if not global_proto_centers_scaled_path:
        return None
    if not os.path.exists(global_proto_centers_scaled_path):
        return None

    cache_key = (
        str(dataset_name).lower(),
        int(step_id),
        str(global_proto_centers_scaled_path),
        str(global_scaler_mean_path),
        str(global_scaler_scale_path),
        str(global_feature_mask),
    )
    if cache_key in downstream_proto_mapper_cache:
        return downstream_proto_mapper_cache[cache_key]

    try:
        mapper = GlobalPrototypeMapper(
            proto_centers_scaled_path=global_proto_centers_scaled_path,
            scaler_mean_path=global_scaler_mean_path,
            scaler_scale_path=global_scaler_scale_path,
            proto_tau=float(args.prototype_assign_tau),
            use_log1p=bool(args.stage1_use_log1p),
            feature_mask=(global_feature_mask or args.stage1_feature_mask),
        )

        g = get_stage1_reference_graph(
            dataset_name=dataset_name,
            data_root=os.path.join(parent_directory, 'data'),
            graph_id=int(step_id),
            remove_selfloops=bool(args.stage1_remove_selfloops),
        )
        subgraphs = split_graph_to_subgraphs(
            g,
            max_nodes=int(args.max_subgraph_nodes),
            hop_n=int(args.stage1_subgraph_hop_n),
        )

        num_nodes = int(data_obj.num_nodes)
        num_proto = int(mapper.num_prototypes)
        node_weights = np.zeros((num_nodes, num_proto), dtype=np.float64)
        node_hits = np.zeros((num_nodes, 1), dtype=np.float64)

        struct_keys = (
            'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
            'var_degree', 'density', 'avg_clustering', 'triangles', 'diameter',
        )

        for sg in subgraphs:
            feat_dict = encode_subgraph(sg)
            feat_vec = np.array([[float(feat_dict[k]) for k in struct_keys]], dtype=np.float64)
            proto_w = mapper.assign(feat_vec)[0].astype(np.float64)
            node_idx = np.array(list(sg.nodes()), dtype=np.int64)
            if node_idx.size == 0:
                continue
            node_weights[node_idx] += proto_w
            node_hits[node_idx] += 1.0

        covered = node_hits[:, 0] > 0
        if covered.any():
            node_weights[covered] = node_weights[covered] / node_hits[covered]

        # 未覆盖节点做均匀分布兜底，避免全零向量导致后续归一化异常。
        if (~covered).any():
            node_weights[~covered] = 1.0 / float(max(1, num_proto))

        row_sum = node_weights.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
        node_weights = (node_weights / row_sum).astype(np.float32)

        downstream_proto_mapper_cache[cache_key] = node_weights
        print(
            f'[struct-router] realtime global prototype weights prepared: '
            f'dataset={dataset_name} graph={int(step_id)} '
            f'nodes={num_nodes} K={num_proto}'
        )
        return node_weights
    except Exception as e:
        print(f'[struct-router] warn: realtime prototype mapping failed for {dataset_name}: {e}')
        downstream_proto_mapper_cache[cache_key] = None
        return None


# 全局原型中心（scaled空间）路径，构建完全局子图池后才会确定
# 预训练前为 None，构建完成后更新（任务9）
global_proto_centers_scaled_path = None
global_scaler_mean_path = None
global_scaler_scale_path = None
global_feature_mask = None
downstream_proto_mapper_cache = {}


# 若不启用全局子图池，构建完整 stage1/stage2。
# 若启用全局子图池（EGRAPHCL），仍需构建每个数据集的 stage1，
# 以供节点级结构标签读取，避免 [struct] skip ... structural_encodings.csv not found。
if args.global_subgraph_pool != 1:
    ensure_context_artifacts(pretrain_dataset_names, build_stage2=True)
elif args.pretrain_method == 'EGRAPHCL':
    ensure_context_artifacts(pretrain_dataset_names, build_stage2=False)

# graphId 转成字符串，方便命名
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

# ------------------- 实验命名 -------------------
# 把关键超参都塞进实验名中，方便区分不同实验
set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
)

# Unified context pipeline 相关命名补充
set_name += f'_ctx_s2_{args.stage2_top_ratio:.2f}_s3'
# 关键链路配置补充，避免不同路由/原型配置互相覆盖 checkpoint
set_name += (
    f'_route_{str(args.prompt_routing_mode).lower()}'
    f'_bank_{str(args.downstream_prompt_bank_mode).lower()}'
    f'_k{int(args.stage1_num_clusters)}'
    f'_enc_{str(args.egraphcl_encoder_mode).lower()}'
    f'_soft{int(args.prototype_soft_prompt)}'
    f'_ptau{float(args.prototype_assign_tau):.2f}'
    f'_pcl{float(args.prototype_consistency_loss_weight):.2f}'
    f'_pbal{float(args.prototype_balance_loss_weight):.2f}'
)

# checkpoint 与 csv 输出路径
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

# 配置日志输出到文件
logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)

# ------------------- 模型构建 -------------------
# PrePrompt 是预训练阶段的核心模型，内部通常包含：
# - GNN 编码器
# - prompt 模块
# - 多数据集融合逻辑
model = PrePrompt(
    unify_dim,
    hid_units,
    nonlinearity,
    num_pretrain_dataset_num,
    int(args.stage1_num_clusters),
    num_layers_num,
    0.1,
    type_=args.combinetype,
    backbone=args.backbone,
    alpha=args.alpha,
    ablation=args.ablation_pre,
).cuda()
model.egraphcl_mask_tau = float(args.egraphcl_tau)
model.egraphcl_mask_strategy = str(args.egraphcl_mask_strategy).lower()
model.egraphcl_mask_quantile = float(args.egraphcl_mask_quantile)
model.egraphcl_mask_min_tau = float(args.egraphcl_mask_min_tau)
model.egraphcl_mask_max_tau = float(args.egraphcl_mask_max_tau)
model.egraphcl_negative_pair_mode = str(args.egraphcl_negative_pair_mode).lower()
model.egraphcl_negative_topk = int(args.egraphcl_negative_topk)
model.egraphcl_negative_random_ratio = float(args.egraphcl_negative_random_ratio)
model.egraphcl_min_negatives_per_row = int(args.egraphcl_min_negatives_per_row)
model.egraphcl_encoder_mode = str(args.egraphcl_encoder_mode).lower()
model.prompt_routing_mode = str(args.prompt_routing_mode).lower()
model.prompt_id_split_base = int(args.prompt_id_split_base)
model.downstream_prompt_bank_mode = str(args.downstream_prompt_bank_mode).lower()
model.egraphcl_log_every = int(args.egraphcl_log_every)
model.egraphcl_batched_chunk_graphs = int(args.egraphcl_batched_chunk_graphs)
model.use_prototype_soft_prompt = int(args.prototype_soft_prompt) == 1
model.prototype_assign_tau = float(args.prototype_assign_tau)
model.prototype_consistency_loss_weight = float(args.prototype_consistency_loss_weight)
model.prototype_balance_loss_weight = float(args.prototype_balance_loss_weight)
# 新增参数传递（任务7）
model.node_loss_weight = float(args.node_loss_weight)
model.egraphcl_core.configure_temperature_schedule(
    schedule=str(args.egraphcl_temperature_schedule).lower(),
    temperature_max=float(args.egraphcl_temperature_max),
    temperature_min=float(args.egraphcl_temperature_min),
)

# 下游评测时，默认把最后 100 个样本作为测试索引
test_idx_num = 100

# 需要参与预训练的图 id
# 注意：这里 graphId 默认是从 1 开始的人类习惯编号
target_graph_id = args.graphId

# ------------------- 尝试跳过预训练，直接加载 checkpoint -------------------
sample_dataset_ids = None
sample_prototype_ids = None
sample_prototype_weights = None

try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))

except Exception as e:
    print(f'[pretrain] checkpoint load skipped due to: {e}')
    # ============================================================
    # 预训练数据准备阶段
    # 主要生成：
    # - feature / adj
    # - augmented feature / adj
    # - label / BCE mask
    # ============================================================

    if args.global_subgraph_pool == 1 and args.pretrain_method == 'EGRAPHCL':
        # graphId 从 1-based 转为 0-based
        target_graph_id_zero_based = [int(gid) - 1 for gid in target_graph_id]

        # 构建全局子图池，并返回被选中的样本及全局 cluster 相似度矩阵
        selected_samples, global_cluster_similarity = build_global_subgraph_pool_records(
            pretrain_dataset_names,
            target_graph_id_zero_based,
        )

        # 缓存整图 feature/adj，避免同一图被重复 load/process
        full_graph_cache = {}
        sample_dataset_ids = []
        sample_prototype_ids = []
        sample_prototype_weights = []
        preprocess_cache_dir = os.path.join(cache_dir, 'egraphcl_preprocess')
        os.makedirs(preprocess_cache_dir, exist_ok=True)

        dataset_name_to_idx = {
            str(name): int(i) for i, name in enumerate(pretrain_dataset_names)
        }

        route_mode = str(args.prompt_routing_mode).lower()

        for sample in selected_samples:
            ds_name = sample['dataset_name']
            ds_idx = sample.get('dataset_idx')
            if ds_idx is None:
                if ds_name not in dataset_name_to_idx:
                    raise KeyError(
                        f"dataset_idx missing and dataset_name={ds_name} not found in pretrain_datasets"
                    )
                ds_idx = dataset_name_to_idx[ds_name]
            graph_id = sample['graph_id']
            node_idx = sample['node_idx']
            cluster_id = sample['cluster_id']
            prototype_id = sample.get('prototype_id', cluster_id)
            prototype_weights = sample.get('prototype_weights', None)
            subgraph_id = int(sample.get('subgraph_id', -1))

            cache_key_name = (
                f"{str(ds_name).lower()}_g{int(graph_id)}_sg{subgraph_id}_"
                f"u{int(unify_dim)}_d{float(drop_percent):.4f}_"
                f"tau{float(args.egraphcl_tau):.4f}_s2r{float(args.stage2_top_ratio):.4f}_"
                f"route{route_mode}_pcawhole.pt"
            )
            sample_cache_path = os.path.join(preprocess_cache_dir, cache_key_name)

            if int(args.pretrain_preprocess_cache) == 1 and os.path.exists(sample_cache_path):
                try:
                    cached = torch_load_trusted(sample_cache_path)
                    feature = cached['feature']
                    adj = cached['adj']
                    aug_feature = cached['aug_feature']
                    aug_adj = cached['aug_adj']
                    lbl = cached['lbl']
                    bce_mask = cached['bce_mask']

                    aug_features.append(aug_feature)
                    aug_adjs.append(aug_adj)
                    lbls.append(lbl)
                    bce_masks.append(bce_mask)
                    features.append(feature)
                    adjs.append(adj)
                    sample_dataset_ids.append(int(ds_idx))
                    sample_prototype_ids.append(int(prototype_id))
                    if prototype_weights is None:
                        fallback = np.zeros(int(args.stage1_num_clusters), dtype=np.float32)
                        pid = int(prototype_id)
                        if 0 <= pid < fallback.shape[0]:
                            fallback[pid] = 1.0
                        sample_prototype_weights.append(fallback)
                    else:
                        sample_prototype_weights.append(np.asarray(prototype_weights, dtype=np.float32))
                    continue
                except Exception as e:
                    print(f'[cache] warn: failed loading preprocess cache {sample_cache_path}: {e}')

            cache_key = (ds_name, graph_id)

            # 若当前整图还没处理过，则先处理整图（并 fit PCA）
            if cache_key not in full_graph_cache:
                ds_obj = load_dataset(ds_name)
                if graph_id < 0 or graph_id >= len(ds_obj):
                    raise IndexError(f'graph_id={graph_id} out of range for dataset={ds_name}')
                data_obj = ds_obj[graph_id]

                # 将图数据转成 feature + sparse adj
                graph_feature, graph_adj = process.process_tu(data_obj, data_obj.x.shape[1])

                # 整图 fit PCA，与基线一致：跨域对齐的关键
                # 每个数据集用整图特征 fit PCA，子图用同一个变换矩阵 transform
                from sklearn.decomposition import PCA as _PCA
                _n_components = min(unify_dim, graph_feature.shape[0], graph_feature.shape[1])
                _pca = _PCA(n_components=_n_components)
                _graph_feature_pca = _pca.fit_transform(graph_feature)
                if _n_components < unify_dim:
                    import numpy as _np
                    _pad = _np.zeros((graph_feature.shape[0], unify_dim - _n_components), dtype=np.float32)
                    _graph_feature_pca = _np.concatenate([_graph_feature_pca, _pad], axis=1)
                _graph_feature_pca = _graph_feature_pca.astype(np.float32)

                full_graph_cache[cache_key] = (graph_feature, graph_adj, _pca, _graph_feature_pca, _n_components)

            graph_feature, graph_adj, graph_pca, graph_feature_pca, pca_n_components = full_graph_cache[cache_key]

            node_idx = np.asarray(node_idx, dtype=np.int64)

            # 按 node_idx 取出子图的 feature / adjacency
            sub_feature_raw = graph_feature[node_idx, :]
            sub_adj = graph_adj[node_idx][:, node_idx]

            # 用整图的 PCA 变换矩阵 transform 子图特征（与基线对齐）
            sub_feature_transformed = graph_pca.transform(sub_feature_raw)
            if pca_n_components < unify_dim:
                _pad = np.zeros((sub_feature_transformed.shape[0], unify_dim - pca_n_components), dtype=np.float32)
                sub_feature_transformed = np.concatenate([sub_feature_transformed, _pad], axis=1)
            feature = torch.FloatTensor(sub_feature_transformed.astype(np.float32))

            # 邻接归一化，并加自环
            adj = process.normalize_adj(sub_adj + sp.eye(sub_adj.shape[0]))

            # 结构标签：任务12 - 尝试获取节点级真实标签，避免整个子图用同一个 id
            # 优先从 StructContextMapper 获取全图节点级标签，再用 node_idx 索引出子图节点的标签
            struct_node_labels = None
            try:
                full_node_labels = struct_context_mapper.get_struct_node_labels(
                    ds_name,
                    int(graph_feature.shape[0]),  # 全图节点数
                    int(graph_id),
                    global_proto_centers_scaled_path=global_proto_centers_scaled_path,
                    global_scaler_mean_path=global_scaler_mean_path,
                    global_scaler_scale_path=global_scaler_scale_path,
                    global_feature_mask=global_feature_mask,
                    proto_tau=args.prototype_assign_tau,
                )
                if full_node_labels is not None:
                    # 用 node_idx 索引出子图节点的标签（节点级，非子图级常数）
                    struct_node_labels = full_node_labels[node_idx]
            except Exception as _e:
                pass

            # 降级：若节点级标签不可用，回退到子图级常数标签
            if struct_node_labels is None:
                route_struct_id = int(prototype_id) if route_mode == 'prototype' else int(cluster_id)
                struct_node_labels = np.full(int(feature.shape[0]), route_struct_id, dtype=np.int64)

            # 构造增强视图、标签和 BCE mask
            aug_feature, aug_adj, lbl, bce_mask = aug.build_aug(
                adj,
                feature,
                sparse,
                drop_percent,
                node_struct_labels=struct_node_labels,
                cluster_similarity=global_cluster_similarity,
                soft_positive_tau=args.egraphcl_tau,
            )

            # 收集预训练所需数据
            aug_features.append(aug_feature)
            aug_adjs.append(aug_adj)
            lbls.append(lbl)
            bce_masks.append(bce_mask)

            features.append(feature)
            adjs.append(adj)
            sample_dataset_ids.append(int(ds_idx))
            sample_prototype_ids.append(int(prototype_id))
            if prototype_weights is None:
                fallback = np.zeros(int(args.stage1_num_clusters), dtype=np.float32)
                pid = int(prototype_id)
                if 0 <= pid < fallback.shape[0]:
                    fallback[pid] = 1.0
                sample_prototype_weights.append(fallback)
            else:
                sample_prototype_weights.append(np.asarray(prototype_weights, dtype=np.float32))

            if int(args.pretrain_preprocess_cache) == 1:
                try:
                    torch.save(
                        {
                            'feature': feature,
                            'adj': adj,
                            'aug_feature': aug_feature,
                            'aug_adj': aug_adj,
                            'lbl': lbl,
                            'bce_mask': bce_mask,
                        },
                        sample_cache_path,
                    )
                except Exception as e:
                    print(f'[cache] warn: failed saving preprocess cache {sample_cache_path}: {e}')

    # ------------------- 优化器 & 数据搬到 GPU -------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

        # feature 张量移动到 GPU
        features = [tensors.cuda() for tensors in features]

        # scipy sparse adjacency 转成 torch sparse 后放到 GPU
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
            for adj in adjs
        ]

        # 各种监督信号 / mask / augment 数据移动到 GPU
        lbls = [tensors.cuda() for tensors in lbls]
        bce_masks = [tensors.cuda() for tensors in bce_masks]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        # LP 模式下，negative_samples 可能为空，此时回退到 negetive_sample
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()

        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ------------------- 预训练循环（带 early stopping） -------------------
    pretrain_batch_size = int(args.pretrain_batch_size)

    # 全局步数计数器（用于温度余弦退火，任务6）
    global_step = 0

    def _slice_list_by_index(items, batch_indices):
        """根据 batch 下标从列表中切片。"""
        return [items[i] for i in batch_indices]

    def _slice_pair_mask(pair_mask, batch_indices):
        """从 pair-wise mask 中切出当前 batch 对应的子矩阵。"""
        if pair_mask is None:
            return None
        idx_t = torch.tensor(batch_indices, dtype=torch.long, device=pair_mask.device)
        return pair_mask.index_select(0, idx_t).index_select(1, idx_t)

    for epoch in range(nb_epochs):
        model.train()
        loss_value = 0.0
        epoch_t0 = time.perf_counter()
        pretrain_forward_sec = 0.0
        pretrain_backward_sec = 0.0

        # 当前仅展示 EGRAPHCL 的预训练逻辑
        if args.pretrain_method == 'EGRAPHCL':
            total_samples = len(aug_features)
            if total_samples <= 0:
                raise RuntimeError('No pretraining samples available for EGRAPHCL.')

            # batch size 不能超过样本数，最少为 1
            effective_bs = max(1, min(pretrain_batch_size, total_samples))

            # 计算总步数（用于温度余弦退火，任务6）
            num_batches_per_epoch = max(1, (total_samples + effective_bs - 1) // effective_bs)
            total_steps = nb_epochs * num_batches_per_epoch

            # 每个 epoch 打乱样本顺序
            perm = torch.randperm(total_samples).tolist()

            epoch_loss_sum = 0.0
            num_batches = 0

            for start in range(0, total_samples, effective_bs):
                batch_indices = perm[start: start + effective_bs]
                batch_t0 = time.perf_counter()

                batch_aug_features = _slice_list_by_index(aug_features, batch_indices)
                batch_aug_adjs = _slice_list_by_index(aug_adjs, batch_indices)
                batch_lbls = _slice_list_by_index(lbls, batch_indices)
                batch_bce_masks = _slice_list_by_index(bce_masks, batch_indices)

                batch_sample_dataset_ids = None
                if sample_dataset_ids is not None:
                    batch_sample_dataset_ids = [sample_dataset_ids[i] for i in batch_indices]

                batch_sample_prototype_ids = None
                if sample_prototype_ids is not None:
                    batch_sample_prototype_ids = [sample_prototype_ids[i] for i in batch_indices]

                batch_sample_prototype_weights = None
                if sample_prototype_weights is not None:
                    # 任务10：只要开了 prototype 路由就传递 weights，
                    # 用于原型正则化（不受 prototype_soft_prompt 控制）。
                    # prototype_soft_prompt 只控制是否在 prompt 混合时使用软权重。
                    if str(args.prompt_routing_mode).lower() == 'prototype' or int(args.prototype_soft_prompt) == 1:
                        batch_sample_prototype_weights = [sample_prototype_weights[i] for i in batch_indices]

                batch_pair_soft_mask = _slice_pair_mask(egraphcl_pair_soft_mask, batch_indices)

                optimiser.zero_grad(set_to_none=True)

                # 前向：PrePrompt 返回当前 batch 的对比损失
                fw_t0 = time.perf_counter()
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
                    sample_prototype_ids=batch_sample_prototype_ids,
                    sample_prototype_weights=batch_sample_prototype_weights,
                    egraphcl_pair_soft_mask=batch_pair_soft_mask,
                    global_step=global_step,
                    total_steps=total_steps,
                )
                global_step += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pretrain_forward_sec += time.perf_counter() - fw_t0

                bw_t0 = time.perf_counter()
                batch_loss.backward()
                optimiser.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pretrain_backward_sec += time.perf_counter() - bw_t0

                epoch_loss_sum += float(batch_loss.detach().item())
                num_batches += 1

                if (num_batches % int(args.egraphcl_log_every)) == 0:
                    extra_stats = {}
                    if hasattr(model, 'latest_egraphcl_stats') and isinstance(model.latest_egraphcl_stats, dict):
                        extra_stats = model.latest_egraphcl_stats

                    if batch_pair_soft_mask is not None:
                        mask_bool = (batch_pair_soft_mask > 0)
                        n = int(mask_bool.shape[0])
                        off_diag_total = max(1, n * (n - 1))
                        mask_ratio = float(mask_bool.sum().item()) / float(off_diag_total)
                    else:
                        mask_ratio = float(extra_stats.get('mask_ratio', -1.0))

                    batch_total_sec = time.perf_counter() - batch_t0
                    print(
                        f'[pretrain][batch] epoch={epoch + 1} '
                        f'batch={num_batches} size={len(batch_indices)} '
                        f'loss={float(batch_loss.detach().item()):.6f} '
                        f"loss_cl={float(extra_stats.get('contrastive_loss', -1.0)):.6f} "
                        f"loss_fea={float(extra_stats.get('fea_loss', -1.0)):.6f} "
                        f"loss_str={float(extra_stats.get('str_loss', -1.0)):.6f} "
                        f"loss_proto_cons={float(extra_stats.get('proto_consistency_loss', 0.0)):.6f} "
                        f"loss_proto_bal={float(extra_stats.get('proto_balance_loss', 0.0)):.6f} "
                        f"loss_proto_reg={float(extra_stats.get('proto_regularization', 0.0)):.6f} "
                        f'mask_ratio={mask_ratio:.6f} '
                        f"tau={float(extra_stats.get('tau', -1.0)):.4f} "
                        f"sim_p90={float(extra_stats.get('sim_p90', 0.0)):.4f} "
                        f"neg_min_fea={int(extra_stats.get('fea_row_neg_min', -1))} "
                        f"neg_min_str={int(extra_stats.get('str_row_neg_min', -1))} "
                        f'batch_sec={batch_total_sec:.4f}'
                    )

            loss_value = epoch_loss_sum / max(1, num_batches)
            print(
                f'[pretrain] epoch={epoch + 1} samples={total_samples} '
                f'batch_size={effective_bs} batches={num_batches}'
            )

        print('Loss:[{:.8f}]'.format(loss_value))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - epoch_t0
        print('[pretrain] epoch_time_sec={:.3f} sec_per_batch={:.4f}'.format(
            epoch_sec,
            epoch_sec / max(1, num_batches) if args.pretrain_method == 'EGRAPHCL' else 0.0,
        ))
        if args.pretrain_method == 'EGRAPHCL':
            print(
                '[pretrain] timing_breakdown_sec '
                f'forward={pretrain_forward_sec:.3f} '
                f'backward={pretrain_backward_sec:.3f} '
                f'other={max(0.0, epoch_sec - pretrain_forward_sec - pretrain_backward_sec):.3f}'
            )

        # ------------------- early stopping -------------------
        # 只有超过 min_delta 的改进才视为有效提升
        if loss_value < (best - pretrain_min_delta):
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

# ============================================================
# 下游评测阶段
# ============================================================
print('#' * 50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)

logging.info('#' * 50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)

# ------------------- 加载下游数据集 -------------------
downstream_dataset = load_dataset(args.dataset)
print(downstream_dataset)

downstream_loader = DataLoader(downstream_dataset)

struct_prompt_router = None

for data in downstream_loader:
    print(data)

    # 将图数据处理成 feature + adjacency
    features, adj = process.process_tu(data, data.x.shape[1])
    print('process done')

    # 特征通过 PCA 压缩到统一维度后放到 GPU
    features = torch.FloatTensor(pca_compression(features, k=unify_dim)).cuda()

    # 邻接归一化并加自环
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    # -------------------------------------------------------
    # 默认取最后 100 个节点作为测试集
    # 注意：这不是标准 benchmark 划分，只是当前实现中的约定
    # -------------------------------------------------------
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y

    # 在下游阶段接线结构原型路由。
    # 优先走“实时结构特征 -> 全局原型软分配”；失败时再回退到 stage1 CSV 标签。
    downstream_stage1_csv = os.path.join(
        stage1_outputs_root,
        args.dataset.lower(),
        'structural_encodings.csv',
    )
    struct_prompt_router = StructPromptRouter(
        StructPromptRouterConfig(
            dataset_name=args.dataset,
            num_prototypes=int(args.stage1_num_clusters),
            graph_step_id=0,
            fallback_graph_step_id=0,
            smoothing_eps=0.0,
        ),
        label_getter=get_struct_node_labels,
        weight_getter=get_struct_node_prototype_weights,
    )
    try:
        prepared = struct_prompt_router.prepare(data)
        if prepared is None:
            struct_prompt_router = None
            print(
                '[struct-router] info: skip structural routing for downstream dataset '
                f'{args.dataset}; realtime/global mapper unavailable and stage1 label fallback unavailable. '
                f'checked fallback csv: {downstream_stage1_csv}'
            )
    except Exception as e:
        struct_prompt_router = None
        print(f'[struct-router] warn: failed to prepare structural routing: {e}')

    # 统计类别数
    label_np = np.array(data.y)
    nb_classes = len(np.unique(label_np))
    print('nb_classes', nb_classes)

    # graph 级任务时，需要为测试集构建子图 batch
    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt

        # graph 分类任务下，构造测试子图
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()

    else:
        from downprompt import downprompt

    # adjacency 转成 torch tensor（稀疏/稠密）
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

# ------------------- 加载最优预训练模型 -------------------
print(f'loading model from {save_name}')
model.load_state_dict(load_state_dict_trusted(save_name))
model = model.cuda()

# 计算下游数据的 embedding
# embeds 通常形状为 [1, num_nodes, hidden_dim]
embeds, _ = model.embed(features, adj, sparse, None, LP)

# 下游分类头学习率候选
downstreamlrlist = [0.001]

# 测试集 embedding（当前主要作备用或调试）
test_embs = embeds[0, idx_test]

# ============================================================
# few-shot 下游评测循环
# 外层遍历下游学习率，中层遍历 shot 数，内层遍历 100 个 split
# ============================================================
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

        # 100 个 few-shot split，对应预先生成的 few-shot 数据划分
        for i in tqdm(range(100)):
            # ------------------- 读取 few-shot 训练数据 -------------------
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
                    "data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    )
                ).type(torch.long).cuda()

                lbls_train = torch.load(
                    "data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i
                    )
                ).type(torch.long).squeeze().cuda()

            # 取出 few-shot 训练节点的预训练 embedding
            pretrain_embs = embeds[0, idx_train]

            # 从预训练模型中取出 prompt / structure / combine 等参数
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights(
                export_bank_mode=args.downstream_prompt_bank_mode
            )

            struct_prototype_weights = None
            if struct_prompt_router is not None:
                struct_prototype_weights = struct_prompt_router.weights_for_tensor(idx_train, features.device)
                if i == 0 and struct_prototype_weights is not None:
                    w = struct_prototype_weights.detach().float()
                    w_safe = torch.clamp(w, min=1e-12)
                    w_entropy = float((-(w_safe * torch.log(w_safe))).sum().item())
                    print(
                        '[struct-router] sample prototype weights ready: '
                        f'shape={tuple(struct_prototype_weights.shape)} '
                        f'sum={float(struct_prototype_weights.sum().item()):.4f} '
                        f'mean={float(w.mean().item()):.6f} '
                        f'max={float(w.max().item()):.6f} '
                        f'entropy={w_entropy:.6f}'
                    )

            # prototype 路由时，用 target 结构权重融合 prompt bank（完成 target-aware 闭环）
            # 任务11：只要有结构原型权重就做混合，不再限制 downstream_prompt_bank_mode=='prototype'
            # 这样默认参数下（auto/dataset 模式）也能利用结构路由器的输出
            if struct_prototype_weights is not None:
                num_banks = len(fea_pretext_weights)
                num_proto_weights = int(struct_prototype_weights.numel())
                if num_proto_weights != int(num_banks):
                    route_mode = str(args.prompt_routing_mode).lower()
                    if route_mode == 'prototype' and int(num_proto_weights) == int(args.stage1_num_clusters):
                        # prototype 路由下，优先显式导出 prototype bank 以匹配 K 维权重。
                        fea_pretext_weights, str_pretext_weights, combines = model.get_weights(
                            export_bank_mode='prototype'
                        )
                        num_banks = len(fea_pretext_weights)

                if int(struct_prototype_weights.numel()) == int(num_banks):
                    proto_w = struct_prototype_weights.detach().float().to(fea_pretext_weights[0].device)
                    proto_w = proto_w / torch.clamp(proto_w.sum(), min=1e-12)

                    mixed_fea = []
                    fea_mix = torch.zeros_like(fea_pretext_weights[0])
                    for bank_id, bank_weight in enumerate(fea_pretext_weights):
                        fea_mix = fea_mix + proto_w[bank_id] * bank_weight
                    mixed_fea.append(fea_mix)

                    mixed_str = []
                    mixed_str_layers = []
                    for layer_idx in range(len(str_pretext_weights[0])):
                        layer_mix = torch.zeros_like(str_pretext_weights[0][layer_idx])
                        for bank_id in range(len(str_pretext_weights)):
                            layer_mix = layer_mix + proto_w[bank_id] * str_pretext_weights[bank_id][layer_idx]
                        mixed_str_layers.append(layer_mix)
                    mixed_str.append(mixed_str_layers)

                    fea_pretext_weights = mixed_fea
                    str_pretext_weights = mixed_str
                elif i == 0:
                    print(
                        '[struct-router] warn: prototype weight dimension mismatch, '
                        f'weights={int(struct_prototype_weights.numel())} banks={int(num_banks)}; '
                        'fallback to unweighted prompt banks.'
                    )

            # 下游阶段额外追加 beta（依赖 downprompt 内部约定）
            combines.append(args.beta)

            # ------------------- 构建下游分类头 -------------------
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

            # 下游分类头优化器
            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            cnt_wait = 0
            best_acc = torch.zeros(1).cuda()

            # ------------------- 在 few-shot 训练集上训练分类头 -------------------
            # 最多 400 step，并以训练 loss 做 early stopping
            for _ in range(400):
                opt.zero_grad()

                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()

                loss = xent(logits, lbls_train)
                loss_value = float(loss.detach().item())

                # early stopping 只看训练 loss
                if loss_value < best:
                    best = loss_value
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    break

                loss.backward()
                opt.step()

            # ------------------- 在测试集上推理 -------------------
            if args.downstream_task == 'graph':
                logits = log(features, adj, sparse, model.gcn, test_index, test_batch)
            else:
                logits = log(features, adj, sparse, model.gcn, idx_test)

            preds = torch.argmax(logits, dim=1).cuda()

            # 准确率
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

            # sklearn 的 F1 计算放到 CPU 上做
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()

            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')

            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)
            tot += acc

        # ------------------- 汇总 100 个 split 的结果 -------------------
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

        # ------------------- 写入 CSV 结果 -------------------
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
                    pretrain_dataset_str,     # 预训练数据集组合
                    downstream_dataset,       # 下游数据集对象
                    acc_mean_formatted,       # 准确率均值
                    acc_std_formatted,        # 准确率标准差
                    microf_mean_formatted,    # micro-F1 均值
                    microf_std_formatted,     # micro-F1 标准差
                    macrof_mean_formatted,    # macro-F1 均值
                    macrof_std_formatted,     # macro-F1 标准差
                ]
            )
