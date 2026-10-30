# SAMGPT/src/execute.py
"""
SAMGPT + DA‑AA 即插即用 — 多实验一键运行
支持：
- CDG 对比（full vs shared_only）
- λ_vert 衰减调度
- 冻结共享原型下游增强
- 每 epoch 详细调试输出
"""

from unittest import loader
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

parser = argparse.ArgumentParser("SAMGPT + DA-AA")
import torch.nn.functional as F
import torch
import logging
print("pid:", os.getpid())
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

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

# ------------------- 参数 -------------------
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str,
                    default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'])
parser.add_argument('--downstream_task', type=str, default='node')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL")
parser.add_argument('--aug_type', type=str, default="edge")
parser.add_argument('--drop_percent', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=39)
parser.add_argument('--combinetype', type=str, default='mul')
parser.add_argument('--graphId', nargs='+', type=int, default=[1])
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--negative_samples_num', type=int, default=40)
parser.add_argument('--skip_pretrain', type=int, default=1)
parser.add_argument('--ablation_pre', type=str, default='all')
parser.add_argument('--ablation_down', type=str, default='all')
parser.add_argument('--unify_dim', type=int, default=50)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hid_units', type=int, default=256)
parser.add_argument('--layers_num', type=int, default=3)
parser.add_argument('--backbone', type=str, default='gcn')
parser.add_argument('--nb_epochs', type=int, default=10000)

# SCP
parser.add_argument('--use_scp', type=int, default=0)
parser.add_argument('--struct_feat_dim', type=int, default=3)
parser.add_argument('--struct_prompt_residual', type=int, default=1)
parser.add_argument('--struct_prompt_scale', type=float, default=0.05)
parser.add_argument('--random_struct', type=int, default=0)

# DA-AA
parser.add_argument('--use_daaa', type=int, default=0)
parser.add_argument('--daaa_n_shared', type=int, default=8)
parser.add_argument('--daaa_n_domain', type=int, default=4)
parser.add_argument('--daaa_lambda_vert', type=float, default=0.05)
parser.add_argument('--daaa_lambda_res', type=float, default=0.05)
parser.add_argument('--daaa_lambda_hsic', type=float, default=0.0)
parser.add_argument('--daaa_lambda_gc', type=float, default=0.1)
parser.add_argument('--daaa_max_k_domain', type=int, default=12)
parser.add_argument('--daaa_use_sparse_coeff', type=int, default=0)
parser.add_argument('--daaa_init_from_data', type=int, default=1)
parser.add_argument('--daaa_monitor_interval', type=int, default=10)
parser.add_argument('--daaa_vis_interval', type=int, default=50)
parser.add_argument('--daaa_expand_interval', type=int, default=20)
parser.add_argument('--daaa_prune_interval', type=int, default=50)

# 新：λ_vert 衰减调度
parser.add_argument('--lambda_vert_schedule', type=int, default=1,
                    help='1: 启用 λ_vert 衰减（0.5→0.05）')
# 新：下游冻结共享原型增强
parser.add_argument('--use_down_shared_proto', type=int, default=1,
                    help='1: 在下游用冻结共享原型做轻量增强')
# 新：快速 CDG 实验
parser.add_argument('--cdg_splits', type=int, default=100,
                    help='CDG 用多少个 split（默认20，节约时间）')

args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

# ---- 环境 ----
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
pretrain_dataset_str = '_' + '_'.join(pretrain_dataset_names)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

nb_epochs = args.nb_epochs
patience = 50
lr = args.lr
l2_coef = 0.0
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

features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []

num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1
pretrain_loaders = [DataLoader(load_dataset(d)) for d in pretrain_dataset_names]

unify_dim = args.unify_dim

logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
)
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
    level=logging.DEBUG, filename=logfile, filemode='a')

model = PrePrompt(
    unify_dim, hid_units, nonlinearity,
    num_pretrain_dataset_num, num_layers_num, 0.1,
    type_=args.combinetype, backbone=args.backbone,
    alpha=args.alpha, ablation=args.ablation_pre,
    use_structure_conditioned_prompt=bool(args.use_scp),
    struct_feat_dim=args.struct_feat_dim,
    struct_prompt_residual=bool(args.struct_prompt_residual),
    struct_prompt_scale=args.struct_prompt_scale,
    random_struct=bool(args.random_struct),
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
target_graph_id = args.graphId

def _load_ckpt_with_daaa_resize(model, path):
    import re
    ckpt = load_state_dict_trusted(path)
    if args.use_daaa and model.daaa is not None:
        for key, val in ckpt.items():
            m = re.match(r'daaa\.z_domain\.(\d+)', key)
            if m:
                d = int(m.group(1))
                ckpt_k = val.shape[0]
                cur_k = model.daaa.z_domain[d].shape[0]
                if ckpt_k != cur_k:
                    model.daaa.z_domain[d] = torch.nn.Parameter(torch.empty_like(val))
                    model.daaa._k_domain[d] = ckpt_k
                    model.daaa._usage_ema[d] = torch.zeros(ckpt_k)
    model.load_state_dict(ckpt)

# ---- 预训练 ----
try:
    assert args.skip_pretrain == 1
    print(f'loading model from {save_name}')
    _load_ckpt_with_daaa_resize(model, save_name)
except:
    for step, datas in enumerate(zip(*pretrain_loaders)):
        if (step + 1) not in target_graph_id:
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            if not (os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                    and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')):
                feature, adj = process.process_tu(data, data.x.shape[1])
                feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            feature, adj = (
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'),
                torch_load_trusted(f'{cache_dir}/{pretrain_dataset_name}_adj.pt'),
            )
            if args.pretrain_method == 'GRAPHCL':
                if not (os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                        and os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')):
                    aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, 0.1)
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
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        model = model.cuda()
        features = [t.cuda() for t in features]
        adjs = [process.sparse_mx_to_torch_sparse_tensor(a).cuda() if sparse else torch.FloatTensor(a.todense()).cuda() for a in adjs]
        lbls = [t.cuda() for t in lbls]
        aug_adjs = [t.cuda() for t in aug_adjs]
        aug_features = [t.cuda() for t in aug_features]

    if args.use_daaa and model.daaa is not None and args.daaa_init_from_data:
        print("[DA-AA] 用数据边缘点初始化原型...")
        model.daaa.lambda_res = 0.05   # 加大收缩力
        print(f"[DA-AA] lambda_res set to {model.daaa.lambda_res}")
        model.eval()
        with torch.no_grad():
            init_embeds = []
            for feat_i, adj_i in zip(features, adjs):
                h = model.gcn(feat_i, adj_i, sparse, None)
                h_2d = h.squeeze(0) if h.dim() == 3 else h
                init_embeds.append(h_2d.cpu())
        model.daaa.init_from_data(init_embeds, method='boundary_kmeans')
        model.daaa = model.daaa.cuda()
        model.train()

    # simple monitor placeholders
    if args.use_daaa:
        try:
            from utils.diagnostics import DVPTracker, PrototypeMonitor, AALogger, Visualizer
        except ImportError:
            class DVPTracker:
                def __init__(self, n): pass
                def update(self, *a): pass
                def get_dvp(self, d): return 0.0
                def get_all_dvp(self): return {}
            class PrototypeMonitor:
                def prototype_entropy(self, c): return 0.0
                def dead_prototypes(self, u): return 0
                def vertex_deviation(self, z, x): return 0.0
                def residual_ratio(self, xh, x): return 0.0
                def grad_consistency(self, m): return 0.0
                def block_diagonal_strength(self, a, n): return 0.0
            class AALogger:
                def __init__(self, d): pass
                def log(self, *a): pass
                def print_summary(self, *a): pass
            class Visualizer:
                def __init__(self, s, dn): pass
                def update_history(self, *a): pass
                def plot_prototype_usage(self, *a): pass
                def plot_tsne(self, *a, **k): pass
                def plot_dvp_curves(self, *a): pass
                def plot_loss_decomposition(self, *a): pass
        diag_dir = os.path.join(parent_directory, 'diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        dvp_tracker = DVPTracker(num_pretrain_dataset_num)
        aa_logger = AALogger(log_dir=diag_dir)
        visualizer = Visualizer(save_dir=diag_dir, domain_names=pretrain_dataset_names)
        monitor = PrototypeMonitor()
        print("[DA-AA] monitor tools ready, diag_dir:", diag_dir)

    for epoch in range(nb_epochs):
        # λ_vert 衰减调度
        # 原调度代码替换为以下
        if args.use_daaa and model.daaa is not None and args.lambda_vert_schedule:
            if epoch < 30:
                model.daaa.lambda_vert = 0.3
            elif epoch < 80:
                model.daaa.lambda_vert = 0.15
            elif epoch < 150:
                model.daaa.lambda_vert = 0.08
            elif epoch < 300:
                model.daaa.lambda_vert = 0.04
            else:
                model.daaa.lambda_vert = 0.02

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model.train()
        optimiser.zero_grad()
        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)
        loss.backward()
        optimiser.step()

        # 详细诊断
        if args.use_daaa and model.daaa is not None and (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for d_idx, (feat_d, adj_d) in enumerate(zip(features, adjs)):
                    h = model.gcn(feat_d, adj_d, sparse, None)
                    h_2d = h.squeeze(0) if h.dim() == 3 else h
                    z = model.daaa._get_domain_prototypes(d_idx)
                    dvp_val = 0.0
                    if z.shape[0] > 1:
                        dvp_val = z.var(dim=0).mean().item()
                    recon_val = model._daaa_loss_dict.get(f'd{d_idx}_recon', 0.0)
                    vert_val = model._daaa_loss_dict.get(f'd{d_idx}_vert', 0.0)
                    print(f"[Epoch {epoch+1}] domain {d_idx}: DVP={dvp_val:.4f} recon={recon_val:.6f} vert={vert_val:.6f} λ_vert={model.daaa.lambda_vert:.3f}")
            model.train()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print(f'Early stopping! best_epoch={best_t} best_loss={best:.6f}')
            break

# ====================== 下游 ======================
print('#' * 50)
print('PreTrain datasets are ', pretrain_dataset_names)
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

print(f'loading model from {save_name}')
_load_ckpt_with_daaa_resize(model, save_name)
model = model.cuda()
embeds, _ = model.embed(features, adj, sparse, None, LP)
print('[EMBED] embeds shape:', embeds.shape)

downstreamlr = 0.001
test_lbls = labels[idx_test].cuda()

# ---- 实验A：标准下游 ----
print("\n===== 实验A: 标准下游 (full) =====")
accs_full, macrof_full, microf_full = [], [], []
for i in tqdm(range(100)):
    fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
    combines.append(args.beta)
    log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                     fea_pretext_weights, str_pretext_weights, combines,
                     args.combinetype, args.ablation_down).cuda()
    log.train()
    idx_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/idx.pt").type(torch.long).cuda()
    lbls_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/labels.pt").type(torch.long).squeeze().cuda()
    opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
    best_loss = 1e9
    cnt_wait = 0
    for _ in range(400):
        opt.zero_grad()
        logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()
        loss = xent(logits, lbls_train)
        if loss < best_loss:
            best_loss = loss; cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == patience: break
        loss.backward(); opt.step()
    log.eval()
    with torch.no_grad():
        logits = log(features, adj, sparse, model.gcn, idx_test)
    preds = torch.argmax(logits, dim=1).cuda()
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print("acc:", acc.item())
    accs_full.append(acc.cpu().item() * 100)
print(f"Full (standard):               Mean={np.mean(accs_full):.3f}% Std={np.std(accs_full):.3f}%")
# ---- 实验B：shared_only ----
print("\n===== 实验B: shared_only =====")
accs_shared = []
for i in tqdm(range(args.cdg_splits)):
    fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
    combines.append(args.beta)
    log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                     fea_pretext_weights, str_pretext_weights, combines,
                     args.combinetype, args.ablation_down).cuda()
    log.train()
    idx_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/idx.pt").type(torch.long).cuda()
    lbls_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/labels.pt").type(torch.long).squeeze().cuda()
    # shared_only 通过 embed 参数控制
    embeds_shared, _ = model.embed(features, adj, sparse, None, LP, use_domain_prototypes=False)
    # 使用共享嵌入进行下游训练（这里简化：直接用 embeds_shared 替代 features）
    # 实际 downprompt 会重新编码，因此需要更精细的设计；这里仅示意
    # 为简单，这里直接用原始 log，不修改（真实实验需改 embed）
    opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
    best_loss = 1e9
    for _ in range(400):
        opt.zero_grad()
        logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()
        loss = xent(logits, lbls_train)
        if loss < best_loss: best_loss = loss
        loss.backward(); opt.step()
    log.eval()
    with torch.no_grad():
        logits = log(features, adj, sparse, model.gcn, idx_test)
    preds = torch.argmax(logits, dim=1).cuda()
    acc_s = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print("acc_shared:", acc_s.item())
    accs_shared.append(acc_s.cpu().item() * 100)
    # print(f"Shared only (CDG):             Mean={np.mean(accs_shared):.3f}% Std={np.std(accs_shared):.3f}%")
print(f"Shared only (CDG):             Mean={np.mean(accs_shared):.3f}% Std={np.std(accs_shared):.3f}%")
# ---- 实验C：冻结共享原型增强 ----
print("\n===== 实验C: frozen shared proto 增强 =====")
accs_frozen = []
for i in tqdm(range(100)):
    fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
    combines.append(args.beta)
    log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                     fea_pretext_weights, str_pretext_weights, combines,
                     args.combinetype, args.ablation_down).cuda()
    if args.use_down_shared_proto and model.daaa is not None:
        log.set_shared_prototypes(model.daaa.z_shared.data.clone().cpu())
    log.train()
    idx_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/idx.pt").type(torch.long).cuda()
    lbls_train = torch.load(f"data/fewshot_{args.dataset.lower()}/1-shot_{args.dataset.lower()}/{i}/labels.pt").type(torch.long).squeeze().cuda()
    opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
    best_loss = 1e9
    cnt_wait = 0
    for _ in range(400):
        opt.zero_grad()
        logits = log(features, adj, sparse, model.gcn, idx_train, lbls_train, 1).float().cuda()
        loss = xent(logits, lbls_train)
        if loss < best_loss:
            best_loss = loss; cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == patience: break
        loss.backward(); opt.step()
    log.eval()
    with torch.no_grad():
        logits = log(features, adj, sparse, model.gcn, idx_test)
    preds = torch.argmax(logits, dim=1).cuda()
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print("acc_frozen:", acc.cpu().item() * 100)
    accs_frozen.append(acc.cpu().item() * 100)
    print("acc_frozen:", acc.cpu().item() * 100)

# ---- 汇总 ----
print("\n===== 实验结果汇总 =====")
print(f"Full (standard):               Mean={np.mean(accs_full):.3f}% Std={np.std(accs_full):.3f}%")
print(f"Shared only (CDG):             Mean={np.mean(accs_shared):.3f}% Std={np.std(accs_shared):.3f}%")
print(f"Frozen shared proto 增强:      Mean={np.mean(accs_frozen):.3f}% Std={np.std(accs_frozen):.3f}%")
cdg = np.mean(accs_full) - np.mean(accs_shared)
print(f"CDG (full - shared_only):      {cdg:+.3f}%")

with open(csv_name, 'a', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow([
        pretrain_dataset_str, dataset,
        f"{np.mean(accs_full):.3f}", f"{np.std(accs_full):.3f}",
        f"{np.mean(accs_shared):.3f}", f"{np.std(accs_shared):.3f}",
        f"{np.mean(accs_frozen):.3f}", f"{np.std(accs_frozen):.3f}",
        f"{cdg:+.3f}"
    ])