# SAMGPT/src/execute.py
"""
SAMGPT + DA‑AA 几何引导预训练与跨域少样本评估
==============================================
主要功能：
1. 多源域图预训练（GRAPHCL / LP）
2. 域感知原型分析 (DA‑AA) 正则化
3. 双级优化几何引导（方案三）
4. 动态原型管理（扩展/修剪/HSIC解耦）
5. 下游 few‑shot 节点/图分类评测（含下游 DA‑AA 增强）
6. 诊断与可视化监控（DVP, 原型熵, t‑SNE 等）

用法示例：
python execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --use_daaa \
    --lambda_geom 0.01 \
    --skip_pretrain 0
"""

from unittest import loader  # noqa (原有，保留兼容)
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import torch.nn as nn
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
# 解析参数
# ------------------------------------------------------------
parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging

# stdout/stderr 实时冲刷（便于 nohup 使用）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# PyTorch 2.6+ 兼容：torch.load 显式设置 weights_only=False
def torch_load_trusted(path: str, map_location=None):
    import inspect
    sig = inspect.signature(torch.load)
    if 'weights_only' in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    else:
        return torch.load(path, map_location=map_location)

def load_state_dict_trusted(path: str, map_location=None):
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd

torch.cuda.empty_cache()

# ---- 基础参数 ----
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str,
                    default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'],
                    help='pretrain datasets')
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

# ---- DA-AA 参数 ----
parser.add_argument('--use_daaa', action='store_true', help='Enable DA-AA')
parser.add_argument('--daaa_n_shared', type=int, default=4)
parser.add_argument('--daaa_n_domain', type=int, default=2)
parser.add_argument('--daaa_lambda_vert', type=float, default=0.05)
parser.add_argument('--daaa_lambda_res', type=float, default=0.01)
parser.add_argument('--max_k', type=int, default=12)
parser.add_argument('--daaa_lambda_hsic', type=float, default=0.0, help='HSIC strength per domain')
parser.add_argument('--daaa_lambda_gc', type=float, default=0.0, help='gradient cosine penalty (experimental)')
parser.add_argument('--daaa_max_k_domain', type=int, default=12, help='max prototypes per domain')
parser.add_argument('--daaa_use_sparse_coeff', action='store_true', help='use sparse coefficient regularization')
# 几何引导
parser.add_argument('--lambda_geom', type=float, default=0.01, help='Geometry guidance strength')
parser.add_argument('--inner_steps', type=int, default=2, help='Inner loop steps')
parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
# 下游 DA-AA
parser.add_argument('--use_daaa_down', action='store_true', help='Enable DA-AA in downstream')
# 原型初始化与管理
parser.add_argument('--daaa_init_from_data', action='store_true', default=True, help='Initialize prototypes from data')
parser.add_argument('--daaa_monitor_interval', type=int, default=10, help='Diagnostic monitoring interval')
parser.add_argument('--daaa_vis_interval', type=int, default=50, help='Visualization interval')
parser.add_argument('--daaa_expand_interval', type=int, default=20, help='Prototype expand interval')
parser.add_argument('--daaa_prune_interval', type=int, default=40, help='Prototype prune interval')

args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

# ---- 环境设置 ----
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_' + strs

print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 路径设置
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

# ---- 超参数 ----
nb_epochs = args.nb_epochs
patience = 100 if args.use_daaa else 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
LP = (args.pretrain_method == 'LP')

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'

dataset = args.dataset
device = torch.device("cuda")

best = 1e9
best_t = 0
cnt_wait = 0

num_layers_num = args.layers_num

# 预训练数据容器
features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
negetive_sample = torch.tensor(0.0)

num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1

pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

unify_dim = args.unify_dim

# 文件与目录
logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

# 模型保存名
_daaa_suffix = ''
if args.use_daaa:
    _daaa_suffix = (
        f'_daaa_ks{args.daaa_n_shared}_kd{args.daaa_n_domain}'
        f'_lv{args.daaa_lambda_vert}_lr{args.daaa_lambda_res}'
    )
set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}{_daaa_suffix}'
)
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)

# ---- 自适应原型数量 ----
_daaa_n_domain_per_domain = None
if args.use_daaa:
    _domain_sizes = []
    for _ds_name in pretrain_dataset_names:
        _feat_path = f'{cache_dir}/{_ds_name}_feature.pt'
        if os.path.exists(_feat_path):
            _f = torch_load_trusted(_feat_path)
            _domain_sizes.append(_f.shape[0])
        else:
            _domain_sizes.append(5000)
    if _domain_sizes:
        _min_size = min(_domain_sizes)
        _daaa_n_domain_per_domain = [
            max(args.daaa_n_domain,
                min(args.daaa_max_k_domain,
                    int(args.daaa_n_domain * (s / _min_size) ** 0.5)))
            for s in _domain_sizes
        ]
        print("[DA-AA] adaptive n_domain per domain:", list(zip(pretrain_dataset_names, _daaa_n_domain_per_domain)))

# ---- 构建模型 ----
model = PrePrompt(
    unify_dim, hid_units, nonlinearity,
    num_pretrain_dataset_num, num_layers_num, 0.1,
    type_ = args.combinetype,
    backbone = args.backbone,
    alpha = args.alpha,
    ablation = args.ablation_pre,
    use_daaa = args.use_daaa,
    daaa_n_shared = args.daaa_n_shared,
    daaa_n_domain = args.daaa_n_domain,
    daaa_lambda_vert = args.daaa_lambda_vert,
    daaa_lambda_res = args.daaa_lambda_res,
    daaa_max_k_domain = args.max_k,
    lambda_geom = args.lambda_geom,
    inner_steps = args.inner_steps,
    inner_lr = args.inner_lr
).cuda()

test_idx_num = 100
target_graph_id = args.graphId

# ---- 加载预训练模型（含DA-AA尺寸适配） ----
def _load_ckpt_with_daaa_resize(model, path):
    import re
    ckpt = load_state_dict_trusted(path)
    if args.use_daaa and model.daaa is not None:
        ckpt_has_daaa = any(k.startswith('daaa.') for k in ckpt)
        if not ckpt_has_daaa:
            print("[DA-AA] checkpoint has no daaa keys, loading with strict=False")
            model.load_state_dict(ckpt, strict=False)
            return
        for key, val in ckpt.items():
            m = re.match(r'daaa\.z_domain\.(\d+)', key)
            if m:
                d = int(m.group(1))
                ckpt_k = val.shape[0]
                cur_k  = model.daaa.z_domain[d].shape[0]
                if ckpt_k != cur_k:
                    model.daaa.z_domain[d] = torch.nn.Parameter(torch.empty_like(val))
                    model.daaa._k_domain[d] = ckpt_k
                    model.daaa._usage_ema[d] = torch.zeros(ckpt_k)
    model.load_state_dict(ckpt)

# ---- 预训练 or 加载 ----
try:
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    _load_ckpt_with_daaa_resize(model, save_name)
except:
    # ---- 准备预训练数据 ----
    for step, datas in enumerate(zip(*pretrain_loaders)):
        if (step + 1) not in target_graph_id:
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            if not (
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            ):
                feature, adj = process.process_tu(data, data.x.shape[1])
                feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')

            feature, adj = (
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'),
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_adj.pt'),
            )

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

            if args.pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_sample = torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_samples.append(negetive_sample)

            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

        if args.pretrain_method == 'LP':
            combinedadj = process.combine_dataset_list_sp(adjs)
            negetive_sample = preprompt.prompt_pretrain_sample(combinedadj, args.negative_samples_num)

    # ---- 优化器 & GPU 转移 ----
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        model = model.cuda()
        features = [tensors.cuda() for tensors in features]
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
            for adj in adjs
        ]
        lbls = [tensors.cuda() for tensors in lbls]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ---- DA-AA 初始化 ----
    if args.use_daaa and model.daaa is not None and args.daaa_init_from_data:
        print("[DA-AA] 用数据边缘点初始化原型...")
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
        model.daaa = model.daaa.cuda()
        model.train()

        # HSIC 按域设置
        _hsic_domains = {
            name: idx for idx, name in enumerate(pretrain_dataset_names)
            if name in ('Photo', 'Computers', 'FacebookPagePage')
        }
        for _name, _idx in _hsic_domains.items():
            model.daaa.set_domain_lambda_hsic(_idx, args.daaa_lambda_hsic if args.daaa_lambda_hsic > 0 else 0.001)
            print("[DA-AA] domain {} ({}) HSIC enabled: lambda={}".format(_idx, _name, model.daaa._get_lambda_hsic(_idx)))

    # ---- 监控工具（占位，可替换为你的实现） ----
    if args.use_daaa:
        # 若 utils.diagnostics 存在则导入，否则使用简单打印
        try:
            from utils.diagnostics import DVPTracker, PrototypeMonitor, AALogger, Visualizer
        except ImportError:
            print("[WARN] diagnostics module not found, using placeholder")
            class DVPTracker:
                def __init__(self, n_domains): pass
                def update(self, *args): pass
                def get_dvp(self, d): return 0.0
                def get_all_dvp(self): return {}
            class PrototypeMonitor:
                def prototype_entropy(self, coeff): return 0.0
                def dead_prototypes(self, usage): return 0
                def vertex_deviation(self, z, x): return 0.0
                def residual_ratio(self, x_hat, x): return 0.0
                def grad_consistency(self, means): return 0.0
                def block_diagonal_strength(self, a_list, n_shared): return 0.0
            class AALogger:
                def __init__(self, log_dir): pass
                def log(self, *args): pass
                def print_summary(self, *args): pass
            class Visualizer:
                def __init__(self, save_dir, domain_names): pass
                def update_history(self, *args): pass
                def plot_prototype_usage(self, *args): pass
                def plot_tsne(self, *args, **kwargs): pass
                def plot_dvp_curves(self, *args): pass
                def plot_loss_decomposition(self, *args): pass

        diag_dir = os.path.join(parent_directory, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        dvp_tracker = DVPTracker(n_domains=num_pretrain_dataset_num)
        aa_logger = AALogger(log_dir=diag_dir)
        visualizer = Visualizer(save_dir=diag_dir, domain_names=pretrain_dataset_names)
        monitor = PrototypeMonitor()
        print("[DA-AA] monitor tools ready, diag_dir:", diag_dir)

    # ---- 预训练循环 ----
    for epoch in range(nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        loss = torch.tensor(0.0).cuda()
        model.train()
        optimiser.zero_grad()

        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)
        elif args.pretrain_method in ('LP', 'splitLP'):
            loss = model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)

        loss.backward()
        optimiser.step()

        print("Loss:[{:.8f}]".format(loss))

        # DA-AA 监控与动态管理
        if args.use_daaa and model.daaa is not None:
            model.eval()
            with torch.no_grad():
                feat_list = features if features else aug_features
                adj_list_cur = adjs

                if (epoch + 1) % args.daaa_monitor_interval == 0:
                    domain_metrics = {}
                    a_list_for_block = []
                    domain_means = []
                    for d_idx2, (feat_d2, adj_d2) in enumerate(zip(feat_list, adj_list_cur)):
                        h2 = model.gcn(feat_d2, adj_d2, sparse, None)
                        h2d2 = h2.squeeze(0) if h2.dim() == 3 else h2
                        domain_means.append(h2d2.mean(dim=0).cpu())
                    grad_cos_val = monitor.grad_consistency(domain_means)

                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        z = model.daaa._get_domain_prototypes(d_idx)
                        a = model.daaa._compute_coefficients(h_2d, z)
                        x_hat = a @ z
                        x_out_d = h_2d + 0.3 * (x_hat - h_2d)

                        dvp_tracker.update(d_idx, z.to(h_2d.device), h_2d, a)

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

                    block_strength = monitor.block_diagonal_strength(a_list_for_block, n_shared=model.daaa.n_shared)

                    avg_losses = {
                        'recon': sum(m.get('recon_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'vert': sum(m.get('vert_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'res': sum(m.get('res_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'hsic': sum(m.get('hsic_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                    }
                    visualizer.update_history(epoch + 1, avg_losses, dvp_tracker.get_all_dvp())

                    aa_logger.print_summary(epoch + 1, domain_metrics)
                    print("  block_diagonal_strength: {:.4f}".format(block_strength))

                if (epoch + 1) % args.daaa_vis_interval == 0:
                    for d_idx in range(num_pretrain_dataset_num):
                        usage = model.daaa.get_usage_ema(d_idx)
                        visualizer.plot_prototype_usage(d_idx, usage, epoch + 1)
                    x_list_vis, z_list_vis = [], []
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        x_list_vis.append(h_2d)
                        z_list_vis.append(model.daaa.z_domain[d_idx])
                    visualizer.plot_tsne(x_list_vis, z_list_vis, epoch + 1, z_shared=model.daaa.z_shared)
                    visualizer.plot_dvp_curves(epoch + 1)
                    visualizer.plot_loss_decomposition(epoch + 1)

                if (epoch + 1) % args.daaa_expand_interval == 0:
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        model.daaa.dynamic_expand(d_idx, h_2d)

                if (epoch + 1) % args.daaa_prune_interval == 0:
                    for d_idx in range(num_pretrain_dataset_num):
                        model.daaa.dynamic_prune(d_idx)

            model.train()

        # Early stopping
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping! best_epoch={} best_loss={:.6f}'.format(best_t, best))
            break
        print('Loading {}th epoch'.format(best_t))

# ====================== 下游评测 ======================
# SAMGPT/src/execute.py
"""
SAMGPT + DA‑AA 几何引导预训练与跨域少样本评估
==============================================
主要功能：
1. 多源域图预训练（GRAPHCL / LP）
2. 域感知原型分析 (DA‑AA) 正则化
3. 双级优化几何引导（方案三）
4. 动态原型管理（扩展/修剪/HSIC解耦）
5. 下游 few‑shot 节点/图分类评测（含下游 DA‑AA 增强）
6. 诊断与可视化监控（DVP, 原型熵, t‑SNE 等）

用法示例：
python execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --use_daaa \
    --lambda_geom 0.01 \
    --skip_pretrain 0
"""

from unittest import loader  # noqa (原有，保留兼容)
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import torch.nn as nn
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
# 解析参数
# ------------------------------------------------------------
parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging

# stdout/stderr 实时冲刷（便于 nohup 使用）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# PyTorch 2.6+ 兼容：torch.load 显式设置 weights_only=False
def torch_load_trusted(path: str, map_location=None):
    import inspect
    sig = inspect.signature(torch.load)
    if 'weights_only' in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    else:
        return torch.load(path, map_location=map_location)

def load_state_dict_trusted(path: str, map_location=None):
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd

torch.cuda.empty_cache()

# ---- 基础参数 ----
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str,
                    default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'],
                    help='pretrain datasets')
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

# ---- DA-AA 参数 ----
parser.add_argument('--use_daaa', action='store_true', help='Enable DA-AA')
parser.add_argument('--daaa_n_shared', type=int, default=4)
parser.add_argument('--daaa_n_domain', type=int, default=2)
parser.add_argument('--daaa_lambda_vert', type=float, default=0.05)
parser.add_argument('--daaa_lambda_res', type=float, default=0.01)
parser.add_argument('--max_k', type=int, default=12)
parser.add_argument('--daaa_lambda_hsic', type=float, default=0.0, help='HSIC strength per domain')
parser.add_argument('--daaa_lambda_gc', type=float, default=0.0, help='gradient cosine penalty (experimental)')
parser.add_argument('--daaa_max_k_domain', type=int, default=12, help='max prototypes per domain')
parser.add_argument('--daaa_use_sparse_coeff', action='store_true', help='use sparse coefficient regularization')
# 几何引导
parser.add_argument('--lambda_geom', type=float, default=0.01, help='Geometry guidance strength')
parser.add_argument('--inner_steps', type=int, default=2, help='Inner loop steps')
parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
# 下游 DA-AA
parser.add_argument('--use_daaa_down', action='store_true', help='Enable DA-AA in downstream')
# 原型初始化与管理
parser.add_argument('--daaa_init_from_data', action='store_true', default=True, help='Initialize prototypes from data')
parser.add_argument('--daaa_monitor_interval', type=int, default=10, help='Diagnostic monitoring interval')
parser.add_argument('--daaa_vis_interval', type=int, default=50, help='Visualization interval')
parser.add_argument('--daaa_expand_interval', type=int, default=20, help='Prototype expand interval')
parser.add_argument('--daaa_prune_interval', type=int, default=40, help='Prototype prune interval')

args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

# ---- 环境设置 ----
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_' + strs

print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 路径设置
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

# ---- 超参数 ----
nb_epochs = args.nb_epochs
patience = 100 if args.use_daaa else 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
LP = (args.pretrain_method == 'LP')

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'

dataset = args.dataset
device = torch.device("cuda")

best = 1e9
best_t = 0
cnt_wait = 0

num_layers_num = args.layers_num

# 预训练数据容器
features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
negetive_sample = torch.tensor(0.0)

num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1

pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

unify_dim = args.unify_dim

# 文件与目录
logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

# 模型保存名
_daaa_suffix = ''
if args.use_daaa:
    _daaa_suffix = (
        f'_daaa_ks{args.daaa_n_shared}_kd{args.daaa_n_domain}'
        f'_lv{args.daaa_lambda_vert}_lr{args.daaa_lambda_res}'
    )
set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}{_daaa_suffix}'
)
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)

# ---- 自适应原型数量 ----
_daaa_n_domain_per_domain = None
if args.use_daaa:
    _domain_sizes = []
    for _ds_name in pretrain_dataset_names:
        _feat_path = f'{cache_dir}/{_ds_name}_feature.pt'
        if os.path.exists(_feat_path):
            _f = torch_load_trusted(_feat_path)
            _domain_sizes.append(_f.shape[0])
        else:
            _domain_sizes.append(5000)
    if _domain_sizes:
        _min_size = min(_domain_sizes)
        _daaa_n_domain_per_domain = [
            max(args.daaa_n_domain,
                min(args.daaa_max_k_domain,
                    int(args.daaa_n_domain * (s / _min_size) ** 0.5)))
            for s in _domain_sizes
        ]
        print("[DA-AA] adaptive n_domain per domain:", list(zip(pretrain_dataset_names, _daaa_n_domain_per_domain)))

# ---- 构建模型 ----
model = PrePrompt(
    unify_dim, hid_units, nonlinearity,
    num_pretrain_dataset_num, num_layers_num, 0.1,
    type_ = args.combinetype,
    backbone = args.backbone,
    alpha = args.alpha,
    ablation = args.ablation_pre,
    use_daaa = args.use_daaa,
    daaa_n_shared = args.daaa_n_shared,
    daaa_n_domain = args.daaa_n_domain,
    daaa_lambda_vert = args.daaa_lambda_vert,
    daaa_lambda_res = args.daaa_lambda_res,
    daaa_max_k_domain = args.max_k,
    lambda_geom = args.lambda_geom,
    inner_steps = args.inner_steps,
    inner_lr = args.inner_lr
).cuda()

test_idx_num = 100
target_graph_id = args.graphId

# ---- 加载预训练模型（含DA-AA尺寸适配） ----
def _load_ckpt_with_daaa_resize(model, path):
    import re
    ckpt = load_state_dict_trusted(path)
    if args.use_daaa and model.daaa is not None:
        ckpt_has_daaa = any(k.startswith('daaa.') for k in ckpt)
        if not ckpt_has_daaa:
            print("[DA-AA] checkpoint has no daaa keys, loading with strict=False")
            model.load_state_dict(ckpt, strict=False)
            return
        for key, val in ckpt.items():
            m = re.match(r'daaa\.z_domain\.(\d+)', key)
            if m:
                d = int(m.group(1))
                ckpt_k = val.shape[0]
                cur_k  = model.daaa.z_domain[d].shape[0]
                if ckpt_k != cur_k:
                    model.daaa.z_domain[d] = torch.nn.Parameter(torch.empty_like(val))
                    model.daaa._k_domain[d] = ckpt_k
                    model.daaa._usage_ema[d] = torch.zeros(ckpt_k)
    model.load_state_dict(ckpt)

# ---- 预训练 or 加载 ----
try:
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    _load_ckpt_with_daaa_resize(model, save_name)
except:
    # ---- 准备预训练数据 ----
    for step, datas in enumerate(zip(*pretrain_loaders)):
        if (step + 1) not in target_graph_id:
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            if not (
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            ):
                feature, adj = process.process_tu(data, data.x.shape[1])
                feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')

            feature, adj = (
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'),
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_adj.pt'),
            )

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

            if args.pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_sample = torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_samples.append(negetive_sample)

            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

        if args.pretrain_method == 'LP':
            combinedadj = process.combine_dataset_list_sp(adjs)
            negetive_sample = preprompt.prompt_pretrain_sample(combinedadj, args.negative_samples_num)

    # ---- 优化器 & GPU 转移 ----
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        model = model.cuda()
        features = [tensors.cuda() for tensors in features]
        adjs = [
            process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda()
            for adj in adjs
        ]
        lbls = [tensors.cuda() for tensors in lbls]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ---- DA-AA 初始化 ----
    if args.use_daaa and model.daaa is not None and args.daaa_init_from_data:
        print("[DA-AA] 用数据边缘点初始化原型...")
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
        model.daaa = model.daaa.cuda()
        model.train()

        # HSIC 按域设置
        _hsic_domains = {
            name: idx for idx, name in enumerate(pretrain_dataset_names)
            if name in ('Photo', 'Computers', 'FacebookPagePage')
        }
        for _name, _idx in _hsic_domains.items():
            model.daaa.set_domain_lambda_hsic(_idx, args.daaa_lambda_hsic if args.daaa_lambda_hsic > 0 else 0.001)
            print("[DA-AA] domain {} ({}) HSIC enabled: lambda={}".format(_idx, _name, model.daaa._get_lambda_hsic(_idx)))

    # ---- 监控工具（占位，可替换为你的实现） ----
    if args.use_daaa:
        # 若 utils.diagnostics 存在则导入，否则使用简单打印
        try:
            from utils.diagnostics import DVPTracker, PrototypeMonitor, AALogger, Visualizer
        except ImportError:
            print("[WARN] diagnostics module not found, using placeholder")
            class DVPTracker:
                def __init__(self, n_domains): pass
                def update(self, *args): pass
                def get_dvp(self, d): return 0.0
                def get_all_dvp(self): return {}
            class PrototypeMonitor:
                def prototype_entropy(self, coeff): return 0.0
                def dead_prototypes(self, usage): return 0
                def vertex_deviation(self, z, x): return 0.0
                def residual_ratio(self, x_hat, x): return 0.0
                def grad_consistency(self, means): return 0.0
                def block_diagonal_strength(self, a_list, n_shared): return 0.0
            class AALogger:
                def __init__(self, log_dir): pass
                def log(self, *args): pass
                def print_summary(self, *args): pass
            class Visualizer:
                def __init__(self, save_dir, domain_names): pass
                def update_history(self, *args): pass
                def plot_prototype_usage(self, *args): pass
                def plot_tsne(self, *args, **kwargs): pass
                def plot_dvp_curves(self, *args): pass
                def plot_loss_decomposition(self, *args): pass

        diag_dir = os.path.join(parent_directory, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        dvp_tracker = DVPTracker(n_domains=num_pretrain_dataset_num)
        aa_logger = AALogger(log_dir=diag_dir)
        visualizer = Visualizer(save_dir=diag_dir, domain_names=pretrain_dataset_names)
        monitor = PrototypeMonitor()
        print("[DA-AA] monitor tools ready, diag_dir:", diag_dir)

    # ---- 预训练循环 ----
    for epoch in range(nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        loss = torch.tensor(0.0).cuda()
        model.train()
        optimiser.zero_grad()

        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)
        elif args.pretrain_method in ('LP', 'splitLP'):
            loss = model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)

        loss.backward()
        optimiser.step()

        print("Loss:[{:.8f}]".format(loss))

        # DA-AA 监控与动态管理
        if args.use_daaa and model.daaa is not None:
            model.eval()
            with torch.no_grad():
                feat_list = features if features else aug_features
                adj_list_cur = adjs

                if (epoch + 1) % args.daaa_monitor_interval == 0:
                    domain_metrics = {}
                    a_list_for_block = []
                    domain_means = []
                    for d_idx2, (feat_d2, adj_d2) in enumerate(zip(feat_list, adj_list_cur)):
                        h2 = model.gcn(feat_d2, adj_d2, sparse, None)
                        h2d2 = h2.squeeze(0) if h2.dim() == 3 else h2
                        domain_means.append(h2d2.mean(dim=0).cpu())
                    grad_cos_val = monitor.grad_consistency(domain_means)

                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        z = model.daaa._get_domain_prototypes(d_idx)
                        a = model.daaa._compute_coefficients(h_2d, z)
                        x_hat = a @ z
                        x_out_d = h_2d + 0.3 * (x_hat - h_2d)

                        dvp_tracker.update(d_idx, z.to(h_2d.device), h_2d, a)

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

                    block_strength = monitor.block_diagonal_strength(a_list_for_block, n_shared=model.daaa.n_shared)

                    avg_losses = {
                        'recon': sum(m.get('recon_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'vert': sum(m.get('vert_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'res': sum(m.get('res_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                        'hsic': sum(m.get('hsic_loss', 0) for m in domain_metrics.values()) / max(len(domain_metrics), 1),
                    }
                    visualizer.update_history(epoch + 1, avg_losses, dvp_tracker.get_all_dvp())

                    aa_logger.print_summary(epoch + 1, domain_metrics)
                    print("  block_diagonal_strength: {:.4f}".format(block_strength))

                if (epoch + 1) % args.daaa_vis_interval == 0:
                    for d_idx in range(num_pretrain_dataset_num):
                        usage = model.daaa.get_usage_ema(d_idx)
                        visualizer.plot_prototype_usage(d_idx, usage, epoch + 1)
                    x_list_vis, z_list_vis = [], []
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        x_list_vis.append(h_2d)
                        z_list_vis.append(model.daaa.z_domain[d_idx])
                    visualizer.plot_tsne(x_list_vis, z_list_vis, epoch + 1, z_shared=model.daaa.z_shared)
                    visualizer.plot_dvp_curves(epoch + 1)
                    visualizer.plot_loss_decomposition(epoch + 1)

                if (epoch + 1) % args.daaa_expand_interval == 0:
                    for d_idx, (feat_d, adj_d) in enumerate(zip(feat_list, adj_list_cur)):
                        h = model.gcn(feat_d, adj_d, sparse, None)
                        h_2d = h.squeeze(0) if h.dim() == 3 else h
                        model.daaa.dynamic_expand(d_idx, h_2d)

                if (epoch + 1) % args.daaa_prune_interval == 0:
                    for d_idx in range(num_pretrain_dataset_num):
                        model.daaa.dynamic_prune(d_idx)

            model.train()

        # Early stopping
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping! best_epoch={} best_loss={:.6f}'.format(best_t, best))
            break
        print('Loading {}th epoch'.format(best_t))
# ----------------- few-shot 下游评测循环 -------------------
# ====================== few-shot 下游评测循环 (含 DA-AA 增强) ======================
print('#' * 50)
print('Downstream dataset is ', args.dataset)

downstream_dataset = load_dataset(args.dataset)
downstream_loader = DataLoader(downstream_dataset)
for data in downstream_loader:
    features, adj = process.process_tu(data, data.x.shape[1])
    features = torch.FloatTensor(pca_compression(features, k=unify_dim)).cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y
    nb_classes = len(np.unique(data.y))

    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt

    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

# 加载最优模型
print(f'loading model from {save_name}')
_load_ckpt_with_daaa_resize(model, save_name)
model = model.cuda()

embeds, _ = model.embed(features, adj, sparse, None, LP)
print('[EMBED] embeds shape:', embeds.shape)

downstreamlrlist = [0.001]
for downstreamlr in downstreamlrlist:
    test_lbls = labels[idx_test].cuda()
    for shotnum in range(1, shot_num + 1):
        accs, macrof, microf = [], [], []
        print("shotnum", shotnum)
        for i in tqdm(range(100)):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            combines.append(args.beta)
            log = downprompt(
                hid_units, nb_classes, unify_dim, num_layers_num,
                fea_pretext_weights, str_pretext_weights, combines,
                args.combinetype, args.ablation_down,
            ).cuda()

            # ---------- 下游 DA-AA 增强 ----------
            if args.use_daaa and model.daaa is not None:
                log.enable_daaa(
                    n_shared=args.daaa_n_shared,
                    n_domain=args.daaa_n_domain,
                    lambda_vert=args.daaa_lambda_vert,
                    lambda_res=args.daaa_lambda_res,
                )
                # 继承预训练的共享原型并冻结
                log.daaa.z_shared.data = model.daaa.z_shared.data.clone().cpu()
                log.daaa.z_shared.requires_grad = False
                # 用目标域嵌入初始化域特化原型
                target_embeds = embeds[0].cpu()
                log.daaa.init_from_data([target_embeds])
                log.daaa = log.daaa.cuda()
                log._daaa_initialized = True

            log.train()
            # 加载 few-shot 训练集
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
            log = log.cuda()
            best_loss = 1e9
            cnt_wait = 0
            for __ in range(400):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()
                loss = xent(logits, lbls_train)
                # 加入 DA-AA 辅助损失
                if args.use_daaa and log.use_daaa:
                    loss += 0.1 * log.get_daaa_loss()
                if loss < best_loss:
                    best_loss = loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    break
                loss.backward()
                opt.step()

            log.eval()
            with torch.no_grad():
                if args.downstream_task == 'graph':
                    logits = log(features, adj, sparse, model.gcn, test_index, test_batch)
                else:
                    logits = log(features, adj, sparse, model.gcn, idx_test)
            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')
            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)

        acc_mean = np.mean([a.cpu().item() if torch.is_tensor(a) else a for a in accs])
        acc_std = np.std([a.cpu().item() if torch.is_tensor(a) else a for a in accs])
        print('Mean:[{:.4f}] Std:[{:.4f}]'.format(acc_mean, acc_std))

        # CDG 计算（仅共享原型）
        if args.use_daaa and model.daaa is not None:
            accs_shared = []
            for i in range(20):   # 只跑20个split以节省时间
                log_shared = downprompt(
                    hid_units, nb_classes, unify_dim, num_layers_num,
                    fea_pretext_weights, str_pretext_weights, combines,
                    args.combinetype, args.ablation_down,
                ).cuda()
                log_shared.enable_daaa(n_shared=args.daaa_n_shared, n_domain=0,
                                       lambda_vert=args.daaa_lambda_vert, lambda_res=args.daaa_lambda_res)
                log_shared.daaa.z_shared.data = model.daaa.z_shared.data.clone().cpu()
                log_shared.daaa.z_shared.requires_grad = False
                log_shared.daaa = log_shared.daaa.cuda()
                log_shared._daaa_initialized = True
                idx_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}/{shotnum}-shot_{args.dataset.lower()}/{i}/idx.pt"
                ).type(torch.long).cuda()
                lbls_train = torch.load(
                    f"data/fewshot_{args.dataset.lower()}/{shotnum}-shot_{args.dataset.lower()}/{i}/labels.pt"
                ).type(torch.long).squeeze().cuda()
                opt_s = torch.optim.Adam(log_shared.parameters(), lr=downstreamlr)
                best_s = 1e9
                for __ in range(400):
                    opt_s.zero_grad()
                    logits_s = log_shared(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()
                    loss_s = xent(logits_s, lbls_train)
                    if loss_s < best_s:
                        best_s = loss_s
                    loss_s.backward()
                    opt_s.step()
                log_shared.eval()
                with torch.no_grad():
                    logits_s = log_shared(features, adj, sparse, model.gcn, idx_test)
                preds_s = torch.argmax(logits_s, dim=1).cuda()
                acc_s = torch.sum(preds_s == test_lbls).float() / test_lbls.shape[0]
                accs_shared.append(acc_s * 100)
            acc_shared_mean = np.mean([a.cpu().item() if torch.is_tensor(a) else a for a in accs_shared])
            cdg = acc_mean - acc_shared_mean
            print('[CDG] shared_only={:.3f}%  full={:.3f}%  CDG={:+.3f}%'.format(acc_shared_mean, acc_mean, cdg))

        with open(csv_name, 'a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([
                pretrain_dataset_str, dataset,
                f"{acc_mean:.3f}", f"{acc_std:.3f}",
                f"{np.mean(microf):.3f}", f"{np.std(microf):.3f}",
                f"{np.mean(macrof):.3f}", f"{np.std(macrof):.3f}"
            ])