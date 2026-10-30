import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module=r"dgl\.backend\.pytorch\.sparse")

from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import importlib
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
import copy
from tqdm import tqdm
from usp_sam import ego_subgraph_mask, ego_subgraph_mask_block, _mask_to_csr
from torch_geometric.loader import DataLoader
from utils.dataset import load_dataset
from utils import process, aug

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
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", choices=['GRAPHCL', 'USP', 'LP', 'splitLP'], help='pretraining method: USP, GRAPHCL, LP, or splitLP')
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
parser.add_argument('--progress_detail',type=str,default='True',help='print detailed progress inside each USP epoch',)
parser.add_argument('--progress_blocks',type=str,default='False',help='show block-level progress for large graph subgraph readout; may be noisy',)
parser.add_argument('--neg_exclude_scope',type=str,default='self',choices=['ego', 'self', 'none'],help='negative sampling exclusion scope',)
parser.add_argument('--neg_refresh_interval', type=int, default=5, help='epochs between negative index refresh')
parser.add_argument('--use_structure_token', type=str, default='True', help='whether USP-SAM uses structure prompt layers')
parser.add_argument('--usp_epochs', type=int, default=20, help='minimal USP-SAM pretraining epochs')
parser.add_argument('--usp_down_epochs', type=int, default=100, help='minimal USP-SAM downstream adaptation epochs')
parser.add_argument('--usp_down_log_interval', type=int, default=10, help='log downstream adaptation metrics every N epochs')
parser.add_argument('--downstream_eval_interval', type=int, default=50, help='run downstream validation every N upstream epochs; <=0 disables periodic validation')
parser.add_argument('--usp_eval_protocol',type=str,default='single_split',choices=['single_split', 'samgpt_standard'],help='USP downstream protocol: one split or SAMGPT-style 100 test nodes over num_splits few-shot splits',)
parser.add_argument('--checkpoint_path', type=str, default=None, help='explicit checkpoint path to load (overrides constructed save_name)')
# NOTE: legacy `--epochs` removed; use `--usp_epochs` for pretraining length.
parser.add_argument('--num_splits', type=int, default=100, help='few-shot splits to evaluate; default preserves original SAMGPT behavior')
parser.add_argument('--split_start', type=int, default=0, help='first few-shot split id')
parser.add_argument('--align_type', type=str, default='none', choices=['none', 'placeholder_structural_role'])
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

from utils.execute_helpers import (
    load_state_dict_trusted,
    prepare_downstream_eval_context,
    resolve_device,
    run_downstream_validation,
    str2bool,
    torch_load_trusted,
)


args.original_pretrain_method = args.pretrain_method
runtime_device = resolve_device(args.device, args.gpu)
if runtime_device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# Note: `--source_domains` and `--target_domain` aliases removed.
# Use `--pretrain_datasets` to specify source datasets and `--dataset` for the target.
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
# Legacy `epochs` parameter removed; ensure downstream code uses `usp_epochs`.
# Note: legacy 'SAMGPT' alias removed — use 'GRAPHCL' instead when needed.

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
if current_dir not in sys.path:
    sys.path.append(current_dir)



run_usp_minimal = importlib.import_module('srcgraphprompt.trainers.usp_minimal_impl')._run_usp_minimal_legacy
_sanity_impl = importlib.import_module('srcgraphprompt.evaluators.sanity_calibration_impl')
_sanity_v3_impl = importlib.import_module('srcgraphprompt.evaluators.sanity_calibration_v3_impl')
run_sanity_calibration = _sanity_impl._run_sanity_calibration_legacy
run_sanity_calibration_v3 = _sanity_v3_impl._run_sanity_calibration_v3_legacy
run_formal_batch1 = importlib.import_module('srcgraphprompt.experiments.formal_batch1_impl')._run_formal_batch1_legacy
run_query_objective_ablation = importlib.import_module('srcgraphprompt.experiments.query_objective_ablation_impl')._run_query_objective_ablation_legacy


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
# Use `usp_epochs` as the unified pretraining epoch hyperparameter.
nb_epochs = args.usp_epochs
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

downstream_eval_context = prepare_downstream_eval_context(args, parent_directory, device, unify_dim, sparse)
downstream_eval_interval = args.downstream_eval_interval

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    ckpt_to_load = args.checkpoint_path if args.checkpoint_path is not None else save_name
    print(f'loading model from {ckpt_to_load}')
    model.load_state_dict(load_state_dict_trusted(ckpt_to_load))
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
                    aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent, aug_type=aug_type)
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
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None, aug_type=aug_type)

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

        if downstream_eval_interval > 0 and (epoch + 1) % downstream_eval_interval == 0:
            periodic_msg = f'[EVAL] periodic downstream validation at upstream epoch {epoch + 1}/{nb_epochs}'
            print(periodic_msg)
            logging.info(periodic_msg)
            model.eval()
            try:
                run_downstream_validation(
                    args,
                    model,
                    downstream_eval_context,
                    device,
                    parent_directory,
                    pretrain_dataset_str,
                    shot_num,
                    num_layers_num,
                    hid_units,
                    unify_dim,
                    sparse,
                    csv_name,
                    stage_label=f'upstream_epoch_{epoch + 1}',
                    write_csv=False,
                )
            finally:
                model.train()

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
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

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
            cnt_wait = 0
            best = float('inf')
            best_state = None

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

            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.to(device)

            # 在 few-shot 训练集上训练分类头，并用当前 split 的训练 loss 做 early stopping
            for _ in range(args.usp_down_epochs):
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
                    best = float(loss.detach())
                    cnt_wait = 0
                    best_state = copy.deepcopy(log.state_dict())
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    # print('Early stopping!')
                    break

                loss.backward()
                opt.step()

            if best_state is not None:
                log.load_state_dict(best_state)

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
                'align_type': 'none',
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
