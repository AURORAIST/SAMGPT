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
    """用于加载本地可信缓存/检查点（兼容新旧版本 PyTorch）。"""
    import inspect
    sig = inspect.signature(torch.load)
    if 'weights_only' in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    else:
        return torch.load(path, map_location=map_location)


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
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--nb_epochs', type=int, default=10000, help='max pretraining epochs')
# ------------------- SCP 参数 -------------------
parser.add_argument('--use_scp', type=int, default=0,
                    help='1: Structure-Conditioned Prompt; 0: original SAMGPT')
parser.add_argument('--struct_feat_dim', type=int, default=3,
                    help='structure feature dim: 1=degree, 3=degree+clustering+pagerank')
parser.add_argument('--struct_prompt_residual', type=int, default=1,
                    help='1: p_v = global + scale*MLP(s); 0: p_v = scale*MLP(s)')
parser.add_argument('--struct_prompt_scale', type=float, default=0.05,
                    help='scale factor for node-level prompt offset')
parser.add_argument('--random_struct', type=int, default=0,
                    help='1: shuffle structure features (ablation)')
# ------------------- DA-AA 参数 -------------------
parser.add_argument('--use_daaa', type=int, default=0,
                    help='1: 启用域感知原型分析（DA-AA）模块')
parser.add_argument('--daaa_n_shared', type=int, default=8,
                    help='DA-AA 共享原型数量 K_s')
parser.add_argument('--daaa_n_domain', type=int, default=4,
                    help='DA-AA 每域特化原型数量 K_d（初始值）')
parser.add_argument('--daaa_lambda_vert', type=float, default=0.05,
                    help='DA-AA 顶点损失系数')
parser.add_argument('--daaa_lambda_res', type=float, default=0.01,
                    help='DA-AA 残差幅度惩罚系数')
parser.add_argument('--daaa_lambda_hsic', type=float, default=0.0,
                    help='DA-AA HSIC 解耦损失系数（0=禁用）')
parser.add_argument('--daaa_lambda_gc', type=float, default=0.1,
                    help='DA-AA 梯度一致性损失系数')
parser.add_argument('--daaa_max_k_domain', type=int, default=12,
                    help='DA-AA 每域最大原型数（动态扩展上限）')
parser.add_argument('--daaa_use_sparse_coeff', type=int, default=0,
                    help='1: 对重构系数做 Top-k 稀疏投影')
parser.add_argument('--daaa_init_from_data', type=int, default=1,
                    help='1: 用数据边缘点初始化原型（推荐）')
parser.add_argument('--daaa_monitor_interval', type=int, default=10,
                    help='DA-AA 监控指标输出间隔（epoch）')
parser.add_argument('--daaa_vis_interval', type=int, default=50,
                    help='DA-AA 可视化输出间隔（epoch）')
parser.add_argument('--daaa_expand_interval', type=int, default=20,
                    help='DA-AA 动态扩展检查间隔（epoch）')
parser.add_argument('--daaa_prune_interval', type=int, default=50,
                    help='DA-AA 动态修剪检查间隔（epoch）')
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
# parent_directory := 项目根（src 的上一级）
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

# ------------------- 训练/模型超参 -------------------
nb_epochs = args.nb_epochs
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
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
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
    # SCP
    use_structure_conditioned_prompt=bool(args.use_scp),
    struct_feat_dim=args.struct_feat_dim,
    struct_prompt_residual=bool(args.struct_prompt_residual),
    struct_prompt_scale=args.struct_prompt_scale,
    random_struct=bool(args.random_struct),
    # DA-AA
    use_daaa=bool(args.use_daaa),
    daaa_n_shared=args.daaa_n_shared,
    daaa_n_domain=args.daaa_n_domain,
    daaa_lambda_vert=args.daaa_lambda_vert,
    daaa_lambda_res=args.daaa_lambda_res,
    daaa_lambda_hsic=args.daaa_lambda_hsic,
    daaa_lambda_gc=args.daaa_lambda_gc,
    daaa_max_k_domain=args.daaa_max_k_domain,
    daaa_use_sparse_coeff=bool(args.daaa_use_sparse_coeff),
).cuda()

test_idx_num = 100
# 选择参与预训练的图 id（针对多图数据集/或 loader zip 的第几个图）
target_graph_id = args.graphId

# ------------------- (可选)跳过预训练，直接加载 checkpoint -------------------
def _load_ckpt_with_daaa_resize(model, path):
    """加载 checkpoint，自动处理 DA-AA 动态扩展导致的形状不匹配。"""
    import re
    ckpt = load_state_dict_trusted(path)
    if args.use_daaa and model.daaa is not None:
        for key, val in ckpt.items():
            m = re.match(r'daaa\.z_domain\.(\d+)', key)
            if m:
                d = int(m.group(1))
                ckpt_k = val.shape[0]
                cur_k  = model.daaa.z_domain[d].shape[0]
                if ckpt_k != cur_k:
                    model.daaa.z_domain[d] = torch.nn.Parameter(
                        torch.empty_like(val)
                    )
                    model.daaa._k_domain[d] = ckpt_k
                    # 同步 _usage_ema，避免 forward 时大小不匹配
                    model.daaa._usage_ema[d] = torch.zeros(ckpt_k)
    model.load_state_dict(ckpt)

try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    _load_ckpt_with_daaa_resize(model, save_name)
except:
    # ------------------- 预训练数据准备（特征/邻接/增强/负采样）-------------------
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

            # ----------- GRAPHCL 方式：需要两份增强视图与对比标签 -----------
            if args.pretrain_method == 'GRAPHCL':
                if not (
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                ):
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
        model = model.cuda()

        # features: Tensor -> cuda
        features = [tensors.cuda() for tensors in features]
        # adjs: scipy sparse -> torch sparse -> cuda
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
            for adj in adjs
        ]
        lbls = [tensors.cuda() for tensors in lbls]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        # LP 情况下 negetive_samples 可能为空，此时使用合并图上的 negetive_sample
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ------------------- DA-AA 初始化（数据边缘点）-------------------
    if args.use_daaa and model.daaa is not None and args.daaa_init_from_data:
        print("[DA-AA] 用数据边缘点初始化原型...")
        # 用 GNN 编码器提取初始嵌入（无 prompt）
        model.eval()
        with torch.no_grad():
            init_embeds = []
            feat_list = features if features else aug_features
            adj_list_init = adjs
            for feat_i, adj_i in zip(feat_list, adj_list_init):
                h = model.gcn(feat_i, adj_i, sparse, None)
                h_2d = h.squeeze(0) if h.dim() == 3 else h
                init_embeds.append(h_2d.cpu())
        model.daaa.init_from_data(init_embeds, method='boundary_kmeans')
        # 初始化完成后将原型迁移到 GPU（与模型保持一致）
        model.daaa = model.daaa.cuda()
        model.train()

    # ------------------- DA-AA 监控工具初始化 -------------------
    if args.use_daaa:
        from utils.diagnostics import (
            DVPTracker, PrototypeMonitor, AALogger, Visualizer
        )
        diag_dir = os.path.join(parent_directory, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        dvp_tracker = DVPTracker(n_domains=num_pretrain_dataset_num)
        aa_logger = AALogger(log_dir=diag_dir)
        visualizer = Visualizer(save_dir=diag_dir, domain_names=pretrain_dataset_names)
        monitor = PrototypeMonitor()
        print("[DA-AA] monitor tools ready, diag_dir: {}".format(diag_dir))

    # ------------------- 预训练循环（early stopping）-------------------
    for epoch in range(nb_epochs):
        # 这里每轮都重设 seed，保证增强/负采样等随机过程在 epoch 间可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        loss = 0
        model.train()
        optimiser.zero_grad()

        # 注册梯度 hook（在 backward 前注册，收集共享原型梯度用于一致性监控）
        if args.use_daaa and model.daaa is not None:
            model.daaa.register_grad_hooks()

        # GRAPHCL: 传入 (aug_features, aug_adjs, lbls)
        if args.pretrain_method == 'GRAPHCL':
            loss = model(
                aug_features, aug_adjs, sparse, None, None, None, lbls, None,
            )

        # LP / splitLP: 传入负采样 samples
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss = model(
                features, adjs, sparse, None, None, None, None, samples=negetive_samples,
            )

        loss.backward()
        optimiser.step()

        print("Loss:[{:.8f}]".format(loss))

        # ------------------- DA-AA 监控与动态管理 -------------------
        if args.use_daaa and model.daaa is not None:
            model.eval()
            with torch.no_grad():
                feat_list = features if features else aug_features
                adj_list_cur = adjs

                # 每 monitor_interval epoch 计算并记录指标
                if (epoch + 1) % args.daaa_monitor_interval == 0:
                    domain_metrics = {}
                    a_list_for_block = []

                    # 计算梯度一致性（当前架构下所有域共享一次backward，无法拆分，标记为-1）
                    grad_cos_val = -1.0

                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h

                        z = model.daaa._get_domain_prototypes(d_idx)
                        a = model.daaa._compute_coefficients(h_2d, z)
                        x_hat = a @ z
                        x_out_d = h_2d + 0.3 * (x_hat - h_2d)

                        # 更新 DVP（传入系数 a，用真实重构计算）
                        z_dom = model.daaa.z_domain[d_idx]
                        dvp_tracker.update(d_idx, z.to(h_2d.device), h_2d, a)

                        # 计算各指标
                        entropy = monitor.prototype_entropy(a)
                        usage_ema = model.daaa.get_usage_ema(d_idx)
                        dead = monitor.dead_prototypes(usage_ema)
                        vert_dev = monitor.vertex_deviation(z, h_2d)
                        res_ratio = monitor.residual_ratio(x_out_d, h_2d)
                        dvp = dvp_tracker.get_dvp(d_idx)

                        a_list_for_block.append(a)

                        metrics = {
                            'dvp': dvp,
                            'recon_loss': model._daaa_loss_dict.get(f'd{d_idx}_recon', 0.0),
                            'vert_loss': model._daaa_loss_dict.get(f'd{d_idx}_vert', 0.0),
                            'res_loss': model._daaa_loss_dict.get(f'd{d_idx}_res', 0.0),
                            'hsic_loss': model._daaa_loss_dict.get(f'd{d_idx}_hsic', 0.0),
                            'entropy': entropy,
                            'dead_protos': dead,
                            'vertex_dev': vert_dev,
                            'res_ratio': res_ratio,
                            'grad_cos': grad_cos_val,
                        }
                        domain_metrics[d_idx] = metrics
                        aa_logger.log(epoch + 1, d_idx, metrics)

                    # 块对角强度
                    block_strength = monitor.block_diagonal_strength(
                        a_list_for_block, n_shared=model.daaa.n_shared
                    )

                    # 更新可视化历史
                    avg_losses = {
                        'recon': sum(m.get('recon_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'vert': sum(m.get('vert_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'res': sum(m.get('res_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'hsic': sum(m.get('hsic_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                    }
                    visualizer.update_history(epoch + 1, avg_losses, dvp_tracker.get_all_dvp())

                    # 打印摘要
                    aa_logger.print_summary(epoch + 1, domain_metrics)
                    print("  block_diagonal_strength: {:.4f}".format(block_strength))

                # 每 vis_interval epoch 生成可视化图表
                if (epoch + 1) % args.daaa_vis_interval == 0:
                    # 原型使用率直方图
                    for d_idx in range(num_pretrain_dataset_num):
                        usage = model.daaa.get_usage_ema(d_idx)
                        visualizer.plot_prototype_usage(d_idx, usage, epoch + 1)

                    # t-SNE 嵌入图（传入共享原型）
                    x_list_vis = []
                    z_list_vis = []
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        x_list_vis.append(h_2d)
                        z_list_vis.append(model.daaa.z_domain[d_idx])
                    visualizer.plot_tsne(x_list_vis, z_list_vis, epoch + 1,
                                         z_shared=model.daaa.z_shared)

                    # DVP 曲线
                    visualizer.plot_dvp_curves(epoch + 1)

                    # 损失分解图
                    visualizer.plot_loss_decomposition(epoch + 1)

                # 每 expand_interval epoch 动态扩展原型
                if (epoch + 1) % args.daaa_expand_interval == 0:
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        model.daaa.dynamic_expand(d_idx, h_2d)

                # 每 prune_interval epoch 动态修剪原型
                if (epoch + 1) % args.daaa_prune_interval == 0:
                    for d_idx in range(num_pretrain_dataset_num):
                        model.daaa.dynamic_prune(d_idx)

            model.train()

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
_load_ckpt_with_daaa_resize(model, save_name)
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
            best_acc = torch.zeros(1).cuda()

            # 在 few-shot 训练集上训练分类头，最多 400 step，并用 loss 做 early stopping
            for _ in range(400):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()

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