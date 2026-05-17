from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
import random

from models import LogReg
from preprompt import PrePrompt, pca_compression
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downprompt, downprompt_graph
import csv
from tqdm import tqdm
import matplotlib
import torch_scatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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


def collect_negative_similarity_distribution(embeddings, labels, exact_pairs_limit=4_000_000, sampled_pairs=2_000_000):
    """按标签是否相同，把节点对分成假负例和真负例，并返回余弦相似度数组。

    为避免大图在 GPU 上构造全部 O(N^2) 节点对导致 OOM：
    - 小图：在 CPU 上做精确统计
    - 大图：在 CPU 上随机采样节点对估计分布
    """
    embeddings = F.normalize(embeddings.detach().cpu(), p=2, dim=1)
    labels = labels.view(-1).detach().cpu()

    num_nodes = labels.shape[0]
    total_pairs = num_nodes * (num_nodes - 1) // 2

    if total_pairs <= exact_pairs_limit:
        pair_index = torch.triu_indices(num_nodes, num_nodes, offset=1)
        pair_similarity = torch.sum(embeddings[pair_index[0]] * embeddings[pair_index[1]], dim=1)
        same_label_mask = labels[pair_index[0]] == labels[pair_index[1]]

        false_negative_similarity = pair_similarity[same_label_mask].numpy()
        true_negative_similarity = pair_similarity[~same_label_mask].numpy()
        return false_negative_similarity, true_negative_similarity

    sample_size = min(sampled_pairs, total_pairs)
    i = torch.randint(0, num_nodes, (sample_size,), dtype=torch.long)
    j = torch.randint(0, num_nodes, (sample_size,), dtype=torch.long)

    valid = i != j
    i = i[valid]
    j = j[valid]
    if i.numel() == 0:
        return np.array([]), np.array([])

    pair_similarity = torch.sum(embeddings[i] * embeddings[j], dim=1)
    same_label_mask = labels[i] == labels[j]

    false_negative_similarity = pair_similarity[same_label_mask].numpy()
    true_negative_similarity = pair_similarity[~same_label_mask].numpy()
    return false_negative_similarity, true_negative_similarity


def _to_numpy_embeddings(embeddings):
    return F.normalize(embeddings.detach().cpu(), p=2, dim=1).numpy()


def _fit_clusterer(embeddings, num_clusters, method, seed):
    x = _to_numpy_embeddings(embeddings)
    cluster_num = max(1, min(int(num_clusters), x.shape[0]))

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=cluster_num, random_state=seed, n_init=10)
        cluster_ids = clusterer.fit_predict(x)
        centers = clusterer.cluster_centers_
    elif method == 'gmm':
        clusterer = GaussianMixture(
            n_components=cluster_num,
            random_state=seed,
            covariance_type='full',
            reg_covar=1e-6,
        )
        cluster_ids = clusterer.fit_predict(x)
        centers = clusterer.means_
    else:
        raise ValueError(f'Unsupported cluster method: {method}')

    return clusterer, cluster_ids, centers, x


def _class_prototypes(embeddings, labels, num_classes):
    labels_np = labels.detach().cpu().view(-1).numpy().astype(np.int64)
    embeds_cpu = embeddings.detach().cpu()
    prototypes = []

    if embeds_cpu.shape[0] == 0:
        raise ValueError('Cannot build class prototypes from empty embeddings')

    fallback = embeds_cpu.mean(dim=0)
    for class_id in range(num_classes):
        mask = labels_np == class_id
        if mask.any():
            prototypes.append(embeds_cpu[torch.from_numpy(mask)].mean(dim=0))
        else:
            prototypes.append(fallback)

    return torch.stack(prototypes, dim=0)


def evaluate_cluster_baseline(train_embeddings, train_labels, test_embeddings, test_labels, num_classes, method, seed, num_clusters=0):
    if method == 'none':
        return None

    train_embeddings = train_embeddings.detach()
    test_embeddings = test_embeddings.detach()
    combined_embeddings = torch.cat([train_embeddings, test_embeddings], dim=0)
    cluster_num = num_clusters if num_clusters > 0 else num_classes
    cluster_num = max(num_classes, cluster_num)

    clusterer, all_cluster_ids, centers, _ = _fit_clusterer(combined_embeddings, cluster_num, method, seed)
    train_cluster_ids = all_cluster_ids[: train_embeddings.shape[0]]
    test_cluster_ids = all_cluster_ids[train_embeddings.shape[0] :]

    train_labels_np = train_labels.detach().cpu().view(-1).numpy().astype(np.int64)
    test_labels_np = test_labels.detach().cpu().view(-1).numpy().astype(np.int64)
    global_majority = int(np.bincount(train_labels_np, minlength=num_classes).argmax())

    class_prototypes = _class_prototypes(train_embeddings, train_labels, num_classes)
    class_prototypes = F.normalize(class_prototypes, p=2, dim=1)
    centers_t = torch.as_tensor(centers, dtype=torch.float32)
    centers_t = F.normalize(centers_t, p=2, dim=1)
    fallback_labels = torch.argmax(torch.matmul(centers_t, class_prototypes.T), dim=1).cpu().numpy()

    cluster_to_label = {}
    for cluster_id in range(len(centers)):
        support_mask = train_cluster_ids == cluster_id
        if support_mask.any():
            counts = np.bincount(train_labels_np[support_mask], minlength=num_classes)
            cluster_to_label[cluster_id] = int(counts.argmax())
        else:
            cluster_to_label[cluster_id] = int(fallback_labels[cluster_id]) if cluster_id < len(fallback_labels) else global_majority

    test_preds = np.array([cluster_to_label.get(int(cluster_id), global_majority) for cluster_id in test_cluster_ids], dtype=np.int64)
    acc = float((test_preds == test_labels_np).mean())
    micro_f1 = f1_score(test_labels_np, test_preds, average='micro')
    macro_f1 = f1_score(test_labels_np, test_preds, average='macro')

    return {
        'cluster_ids': all_cluster_ids,
        'train_cluster_ids': train_cluster_ids,
        'test_cluster_ids': test_cluster_ids,
        'cluster_to_label': cluster_to_label,
        'accuracy': acc,
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'num_clusters': len(centers),
        'cluster_method': method,
        'clusterer': clusterer,
    }


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os




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
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", choices=['GRAPHCL'], help='pretrain method')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help="target graph's id in one dataset")
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--enable_cluster_enhance', type=int, default=0, help='enable clustering-enhanced prototype loss in pretraining')
parser.add_argument('--intra_clusters', type=int, default=8, help='number of intra-domain clusters per domain')
parser.add_argument('--shared_prototypes_num', type=int, default=16, help='number of shared prototypes across domains')
parser.add_argument('--cluster_interval', type=int, default=1, help='apply cluster-enhance loss every N pretrain steps')
parser.add_argument('--cluster_tau', type=float, default=0.2, help='temperature for cluster contrastive alignment')
parser.add_argument('--lambda_cross', type=float, default=0.1, help='weight for cross-domain prototype contrastive loss')
parser.add_argument('--lambda_reg', type=float, default=0.01, help='weight for structure token to local prototype regularization')
parser.add_argument('--lambda_proto', type=float, default=0.05, help='weight for local-shared prototype reconstruction loss')
parser.add_argument('--cluster_conf_threshold', type=float, default=0.6, help='confidence threshold for cross-domain prototype matching')
parser.add_argument('--prototype_ema_momentum', type=float, default=0.95, help='EMA momentum for shared prototypes update')
parser.add_argument('--enable_similarity_plot', type=int, default=0, help='enable similarity density plot (1 enable / 0 disable)')
parser.add_argument(
    '--downstream_cluster_method',
    type=str,
    default='none',
    choices=['none', 'kmeans', 'gmm'],
    help='optional clustering baseline on downstream embeddings',
)
parser.add_argument(
    '--downstream_cluster_num',
    type=int,
    default=0,
    help='number of clusters for downstream clustering baseline; 0 uses nb_classes',
)
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
nb_epochs = 3000
print("nb_epochs: ", nb_epochs)
patience = 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True

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
lbls = []
pretrain_groundtruth_labels = []

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
cluster_csv_name = os.path.join(result_dir, f'{set_name}_{args.downstream_cluster_method}.csv')

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
    enable_cluster_enhance=(args.enable_cluster_enhance == 1),
    intra_clusters=args.intra_clusters,
    shared_prototypes_num=args.shared_prototypes_num,
    cluster_interval=args.cluster_interval,
    cluster_tau=args.cluster_tau,
    lambda_cross=args.lambda_cross,
    lambda_reg=args.lambda_reg,
    lambda_proto=args.lambda_proto,
    cluster_conf_threshold=args.cluster_conf_threshold,
    prototype_ema_momentum=args.prototype_ema_momentum,
).cuda()

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
    # ------------------- 预训练数据准备（特征/邻接/增强）-------------------
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
            pretrain_groundtruth_labels.append(data.y.detach().clone().view(-1))

            # ----------- GRAPHCL 方式：需要两份增强视图与对比标签 -----------
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

            # 邻接统一做归一化并缓存到列表
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

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
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]

    # ------------------- 预训练循环（early stopping）-------------------
    for epoch in range(nb_epochs):
        # 这里每轮都重设 seed，保证增强/负采样等随机过程在 epoch 间可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        loss = 0
        model.train()
        optimiser.zero_grad()

        # GRAPHCL: 传入 (aug_features, aug_adjs, lbls)
        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls)

        loss.backward()
        optimiser.step()

        loss_breakdown = getattr(model, 'last_loss_breakdown', {})
        cluster_stats = getattr(model, 'last_cluster_stats', [])
        cluster_stats_str = '; '.join(
            [
                f"d{stat.get('domain_idx', '?')}:k={stat.get('num_clusters', 0)}/c={stat.get('num_centers', 0)}/hist={stat.get('cluster_hist', [])}"
                for stat in cluster_stats
            ]
        )
        print(
            'Loss:[{:.8f}] base:[{:.8f}] cluster:[{:.8f}] proto:[{:.8f}] cross:[{:.8f}] reg:[{:.8f}] '
            'doms:[{}] scale:[{:.4f}] ent:[{:.4f}] minr:[{:.4f}] maxr:[{:.4f}] '
            'ari:[{:.4f}] gap:[{:.4f}] cov:[{:.4f}] pcoll:[{:.4f}] dgap:[{:.4f}] cstep:[{}] ctri:[{}] clusters:[{}]'.format(
                float(loss.detach().item()),
                float(loss_breakdown.get('base_loss', 0.0)),
                float(loss_breakdown.get('cluster_loss', 0.0)),
                float(loss_breakdown.get('loss_proto', 0.0)),
                float(loss_breakdown.get('loss_cross', 0.0)),
                float(loss_breakdown.get('loss_reg', 0.0)),
                int(loss_breakdown.get('num_domains', 0)),
                float(loss_breakdown.get('cluster_scale', 1.0)),
                float(loss_breakdown.get('avg_assignment_entropy', 0.0)),
                float(loss_breakdown.get('avg_min_cluster_ratio', 0.0)),
                float(loss_breakdown.get('avg_max_cluster_ratio', 0.0)),
                float(loss_breakdown.get('avg_cluster_ari', 0.0)),
                float(loss_breakdown.get('cross_gap', 0.0)),
                float(loss_breakdown.get('match_coverage', 0.0)),
                float(loss_breakdown.get('proto_collapse', 0.0)),
                float(loss_breakdown.get('domain_gap', 0.0)),
                int(loss_breakdown.get('cluster_step', 0)),
                int(loss_breakdown.get('cluster_triggered', 0)),
                cluster_stats_str,
            )
        )
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
        print('Best checkpoint epoch {}'.format(best_t))


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

    downstream_features, adj = process.process_tu(data, data.x.shape[1])
    print('process done')
    downstream_features = torch.FloatTensor(pca_compression(downstream_features, k=unify_dim)).cuda()

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

def plot_similarity_density(false_negative_similarity, true_negative_similarity, save_path, title):
    """绘制真负例/假负例的相似度 density 曲线（已修复：全局归一化到 TN+FN）"""

    def _kde_curve(values, total_count):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        n = len(values)

        if n == 0:
            return None, None

        # 只有1个点 或 方差极小
        if n == 1 or np.std(values) < 1e-8:
            x_axis = np.linspace(values[0] - 0.05, values[0] + 0.05, 200)
            y_axis = np.zeros_like(x_axis)
            y_axis[len(y_axis) // 2] = n / total_count  # 修复点
            return x_axis, y_axis

        kde = gaussian_kde(values)
        lower = max(0.0, values.min() - 0.05)
        upper = min(1.0, values.max() + 0.05)
        x_axis = np.linspace(lower, upper, 400)
        y_axis = kde(x_axis)

        # -------------------------- 核心修复 --------------------------
        # kde 内部归一化为1 → 现在缩放到 本组数量 / 全局总数
        y_axis = y_axis * (n / total_count)
        # --------------------------------------------------------------

        return x_axis, y_axis

    plt.figure(figsize=(8, 5.5))

    fn_vals = false_negative_similarity
    tn_vals = true_negative_similarity

    # 全局总数：TN + FN
    total_count = len(fn_vals) + len(tn_vals)

    curve_specs = [
        (fn_vals, "False negative", "#d1495b"),
        (tn_vals, "True negative", "#2a9d8f"),
    ]

    for values, label_name, color in curve_specs:
        x_axis, y_axis = _kde_curve(values, total_count)
        if x_axis is None:
            continue
        plt.plot(x_axis, y_axis, linewidth=2.2, label=f"{label_name} (n={len(values)})", color=color)
        plt.fill_between(x_axis, y_axis, alpha=0.16, color=color)

    plt.xlabel("Cosine similarity")
    plt.ylabel(f"Density (normalized by TN+FN={total_count})")  # 更清晰
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_similarity_plot_inputs(dataset_name):
    """独立恢复用于画相似度分布的数据。"""
    dataset_obj = load_dataset(dataset_name)
    data = dataset_obj[0]

    feature_cache_path = os.path.join(cache_dir, f'{dataset_name}_feature.pt')
    adj_cache_path = os.path.join(cache_dir, f'{dataset_name}_adj.pt')

    if os.path.exists(feature_cache_path) and os.path.exists(adj_cache_path):
        feature = torch_load_trusted(feature_cache_path)
        adj = torch_load_trusted(adj_cache_path)
    else:
        feature, adj = process.process_tu(data, data.x.shape[1])
        feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))

    feature = feature.cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

    labels = data.y.detach().clone().view(-1).cuda()
    return feature, adj, labels

# 在每个预训练数据集上分别画真负例/假负例相似度密度图。
if args.enable_similarity_plot:
    for pretrain_dataset_name in pretrain_dataset_names:
        dataset_index = pretrain_dataset_names.index(pretrain_dataset_name)
        model.eval()
        with torch.no_grad():
            if len(features) > dataset_index and len(adjs) > dataset_index and len(pretrain_groundtruth_labels) > dataset_index:
                plot_feature = features[dataset_index]
                plot_adj = adjs[dataset_index]
                plot_labels = pretrain_groundtruth_labels[dataset_index].cuda()
            else:
                plot_feature, plot_adj, plot_labels = load_similarity_plot_inputs(pretrain_dataset_name)

            if plot_feature.dim() == 1:
                raise ValueError(
                    f"Expected 2D node features for similarity plotting, got shape {tuple(plot_feature.shape)}"
                )

            pretrain_embeds, _ = model.embed(plot_feature, plot_adj, sparse, None)
        false_negative_similarity, true_negative_similarity = collect_negative_similarity_distribution(
            pretrain_embeds.squeeze(0),
            plot_labels,
        )
        similarity_plot_path = os.path.join(
            result_dir,
            f'{set_name}_{pretrain_dataset_name.lower()}_false_true_negative_similarity_density.png',
        )
        plot_similarity_density(
            false_negative_similarity,
            true_negative_similarity,
            similarity_plot_path,
            f'{pretrain_dataset_name} pretrain: false vs true negative similarity density',
        )
        print(f'Similarity density plot saved to {similarity_plot_path}')

# model.embed 返回 embedding；embeds[0, idx] 是节点 idx 的表示
embeds, _ = model.embed(downstream_features, adj, sparse, None)

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
        cluster_accs = []
        cluster_macrof = []
        cluster_microf = []

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
                    logits = log(downstream_features, adj, sparse, model.gcn, idx_train, batch_train, lbls_train, 1).float().cuda()
                else:
                    logits = log(downstream_features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()

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
                logits = log(downstream_features, adj, sparse, model.gcn, test_index, test_batch)
            else:
                logits = log(downstream_features, adj, sparse, model.gcn, idx_test)

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

            if args.downstream_cluster_method != 'none':
                if args.downstream_task == 'graph':
                    train_cluster_embeddings = torch_scatter.scatter(
                        src=embeds[0, idx_train],
                        index=batch_train,
                        dim=0,
                        reduce='mean',
                    )
                    test_cluster_embeddings = torch_scatter.scatter(
                        src=embeds[0, test_index],
                        index=test_batch,
                        dim=0,
                        reduce='mean',
                    )
                else:
                    train_cluster_embeddings = embeds[0, idx_train]
                    test_cluster_embeddings = embeds[0, idx_test]

                cluster_eval = evaluate_cluster_baseline(
                    train_cluster_embeddings,
                    lbls_train,
                    test_cluster_embeddings,
                    test_lbls,
                    nb_classes,
                    args.downstream_cluster_method,
                    seed,
                    args.downstream_cluster_num,
                )

                if cluster_eval is not None:
                    cluster_accs.append(cluster_eval['accuracy'] * 100)
                    cluster_microf.append(cluster_eval['micro_f1'] * 100)
                    cluster_macrof.append(cluster_eval['macro_f1'] * 100)

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

        if args.downstream_cluster_method != 'none' and len(cluster_accs) > 0:
            cluster_accs_tensor = torch.tensor(cluster_accs)
            cluster_acc_mean = float(cluster_accs_tensor.mean().item())
            cluster_acc_std = float(cluster_accs_tensor.std().item())
            cluster_microf_mean = float(sum(cluster_microf) / len(cluster_microf))
            cluster_macrof_mean = float(sum(cluster_macrof) / len(cluster_macrof))
            cluster_microf_std = float(torch.std(torch.tensor(cluster_microf)).item())
            cluster_macrof_std = float(torch.std(torch.tensor(cluster_macrof)).item())

            print(
                f'[cluster-{args.downstream_cluster_method}] Mean:[{cluster_acc_mean:.4f}] Std:[{cluster_acc_std:.4f}] '
                f'microF1:[{cluster_microf_mean:.4f}] macroF1:[{cluster_macrof_mean:.4f}]'
            )

            with open(f'{cluster_csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file, dialect='excel')
                if file.tell() == 0:
                    writer.writerow(
                        [
                            'method',
                            'pretrain_datasets',
                            'downstream_dataset',
                            'acc_mean',
                            'acc_std',
                            'microf_mean',
                            'microf_std',
                            'macrof_mean',
                            'macrof_std',
                        ]
                    )

                writer.writerow(
                    [
                        args.downstream_cluster_method,
                        pretrain_dataset_str,
                        downstream_dataset,
                        f'{cluster_acc_mean:.3f}',
                        f'{cluster_acc_std:.3f}',
                        f'{cluster_microf_mean:.3f}',
                        f'{cluster_microf_std:.3f}',
                        f'{cluster_macrof_mean:.3f}',
                        f'{cluster_macrof_std:.3f}',
                    ]
                )

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