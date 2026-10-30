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

parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging

try:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

def torch_load_trusted(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)

def load_state_dict_trusted(path: str, map_location=None):
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
parser.add_argument('--skip_pretrain', type=int, default=0, help='1: skip pretraining and load checkpoint')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')

# SCP 参数
parser.add_argument('--use_structure_conditioned_prompt', type=int, default=0,
                    help='1: use node-level structure-conditioned prompt; 0: original SAMGPT')
parser.add_argument('--struct_feat_dim', type=int, default=1,
                    help='structure feature dimension. 1=degree, 3=deg+clust+pagerank')
parser.add_argument('--struct_prompt_residual', type=int, default=1,
                    help='1: p_v = global_prompt + MLP(structure_feat); 0: p_v = MLP(structure_feat)')
parser.add_argument('--struct_prompt_scale', type=float, default=0.05,
                    help='scale factor for node-conditioned prompt (gamma)')
parser.add_argument('--random_struct', type=int, default=0,
                    help='1: randomly shuffle structure feature, for ablation')

# Routing 与升级参数
parser.add_argument('--use_prompt_routing', type=int, default=0,
                    help='1: use node-wise domain router (downstream)')
parser.add_argument('--num_structure_prompts', type=int, default=8,
                    help='(not used for node-wise, kept for compatibility)')
parser.add_argument('--routing_tau', type=float, default=1.0,
                    help='(not used for node-wise, kept for compatibility)')
parser.add_argument('--freeze_generators', type=int, default=1,
                    help='1: freeze all pretrained generators; 0: unfreeze all')
parser.add_argument('--router_tau', type=float, default=3.0,
                    help='temperature for domain router')
parser.add_argument('--entropy_weight', type=float, default=0.0,
                    help='weight for entropy regularization')

# 新升级控制参数
parser.add_argument('--use_sparse_routing', type=int, default=1,
                    help='1: use sparse top-k routing')
parser.add_argument('--sparse_topk', type=int, default=2,
                    help='top-k value for sparse routing')
parser.add_argument('--use_film', type=int, default=1,
                    help='1: use FiLM affine modulation')
parser.add_argument('--use_subspace_prompt', type=int, default=1,
                    help='1: use subspace prompt basis')
parser.add_argument('--subspace_basis_num', type=int, default=8,
                    help='number of subspace basis vectors')

args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_' + strs

import os
print(os.getpid())
print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn

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

# ------------------- 训练/模型超参 -------------------
nb_epochs = 10000
print("nb_epochs:", nb_epochs)
patience = 50
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
firstbest = 0
cnt_wait = 0

num_layers_num = args.layers_num

features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
negetive_sample = torch.tensor(0.0)

print(pretrain_dataset_names)

num_pretrain_dataset_num = len(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1

pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

unify_dim = args.unify_dim

logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
# 使用独立的缓存目录，避免与之前实验冲突
cache_dir = os.path.join(parent_directory, 'cacherouting_scpp')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'

scp_tag = (
    f'scp{args.use_structure_conditioned_prompt}_'
    f'sfd{args.struct_feat_dim}_'
    f'res{args.struct_prompt_residual}_'
    f'scale{args.struct_prompt_scale:.2f}_'
    f'rand{args.random_struct}'
)
routing_tag = (
    f'_routing{args.use_prompt_routing}_'
    f'K{args.num_structure_prompts}_'
    f'tau{args.routing_tau:.1f}'
) if args.use_prompt_routing else ''

set_name = (
    f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_'
    f'{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_'
    f'{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}_'
    f'{scp_tag}{routing_tag}'
)

save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename=logfile,
    filemode='a',
)

model = PrePrompt(
    unify_dim,
    hid_units,
    nonlinearity,
    num_pretrain_dataset_num,
    num_layers_num,
    drop_prob,
    type_=args.combinetype,
    backbone=args.backbone,
    alpha=args.alpha,
    ablation=args.ablation_pre,
    use_structure_conditioned_prompt=bool(args.use_structure_conditioned_prompt),
    struct_feat_dim=args.struct_feat_dim,
    struct_prompt_residual=bool(args.struct_prompt_residual),
    struct_prompt_scale=args.struct_prompt_scale,
    random_struct=bool(args.random_struct),
    use_prompt_routing=bool(args.use_prompt_routing),
    num_structure_prompts=args.num_structure_prompts,
    routing_tau=args.routing_tau,
).cuda()

test_idx_num = 100
target_graph_id = args.graphId

# ------------------- 预训练或加载 checkpoint -------------------
if args.skip_pretrain == 0:
    # ===== 预训练阶段 =====
    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
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
            print('combinedadj', combinedadj.shape)
            negetive_sample = preprompt.prompt_pretrain_sample(combinedadj, args.negative_samples_num)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        print('Using CUDA')
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

    for epoch in range(nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        loss = 0
        model.train()
        optimiser.zero_grad()
        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss = model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)
        loss.backward()
        optimiser.step()
        print('Loss:[{:.8f}]'.format(loss))
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
else:
    # ===== 直接加载已有 checkpoint =====
    print(f'Skip pretraining, loading model from {save_name}')
    model.load_state_dict(load_state_dict_trusted(save_name))

# ------------------- 下游评测准备 -------------------
print('#' * 50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)

downstream_dataset = load_dataset(args.dataset)
print(downstream_dataset)
downstream_loader = DataLoader(downstream_dataset)

for data in downstream_loader:
    print(data)
    features, adj = process.process_tu(data, data.x.shape[1])
    adj_raw = adj.copy()   # 保存原始邻接
    print('process done')
    features = torch.FloatTensor(pca_compression(features, k=unify_dim)).cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y
    data_arr = np.array(data.y)
    np.unique(data_arr)
    nb_classes = len(np.unique(data_arr))
    print('nb_classes', nb_classes)

    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt_graph_class
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt as downprompt_node

    if sparse:
        adj_t = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
        adj_raw_t = process.sparse_mx_to_torch_sparse_tensor(adj_raw).cuda()
    else:
        adj_t = torch.FloatTensor(adj.todense()).cuda()
        adj_raw_t = torch.FloatTensor(adj_raw.todense()).cuda()

model = model.cuda()
structure_generators = model.get_structure_generators()
embeds, _ = model.embed(features, adj_t, sparse, None, LP)

downstreamlrlist = [0.001]

# ------------------- few-shot 下游评测循环 -------------------
for downstreamlr in downstreamlrlist:
    test_lbls = labels[idx_test].cuda()
    print('-' * 100)
    for shotnum in range(1, shot_num + 1):
        tot = torch.zeros(1).cuda()
        accs, macrof, microf = [], [], []
        cnt_wait = 0
        best = 1e9
        print("shotnum", shotnum)
        global_step = 0

        for i in tqdm(range(100)):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            combines.append(args.beta)

            # 构造下游模型，传入所有新参数
            if args.downstream_task == 'graph':
                log = downprompt_graph_class(
                    hid_units, nb_classes, unify_dim, num_layers_num,
                    fea_pretext_weights, str_pretext_weights, combines,
                    args.combinetype, args.ablation_down,
                    use_structure_conditioned_prompt=bool(args.use_structure_conditioned_prompt),
                    struct_feat_dim=args.struct_feat_dim,
                    struct_prompt_residual=bool(args.struct_prompt_residual),
                    struct_prompt_scale=args.struct_prompt_scale,
                    num_pretrain_datasets=len(pretrain_dataset_names),
                    structure_prompt_generators=structure_generators,
                    freeze_generators=bool(args.freeze_generators),
                    router_tau=args.router_tau,
                    use_sparse_routing=bool(args.use_sparse_routing),
                    sparse_topk=args.sparse_topk,
                    use_film=bool(args.use_film),
                    use_subspace_prompt=bool(args.use_subspace_prompt),
                    subspace_basis_num=args.subspace_basis_num,
                ).cuda()
            else:
                log = downprompt_node(
                    hid_units, nb_classes, unify_dim, num_layers_num,
                    fea_pretext_weights, str_pretext_weights, combines,
                    args.combinetype, args.ablation_down,
                    use_structure_conditioned_prompt=bool(args.use_structure_conditioned_prompt),
                    struct_feat_dim=args.struct_feat_dim,
                    struct_prompt_residual=bool(args.struct_prompt_residual),
                    struct_prompt_scale=args.struct_prompt_scale,
                    num_pretrain_datasets=len(pretrain_dataset_names),
                    structure_prompt_generators=structure_generators,
                    freeze_generators=bool(args.freeze_generators),
                    router_tau=args.router_tau,
                    use_sparse_routing=bool(args.use_sparse_routing),
                    sparse_topk=args.sparse_topk,
                    use_film=bool(args.use_film),
                    use_subspace_prompt=bool(args.use_subspace_prompt),
                    subspace_basis_num=args.subspace_basis_num,
                ).cuda()

            log.train()

            if args.downstream_task == 'graph':
                idx_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).cuda()
                batch_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).cuda()
                lbls_train = torch.load(
                    "data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".format(
                        args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).squeeze().cuda()
            else:
                idx_train = torch.load(
                    "data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).cuda()
                lbls_train = torch.load(
                    "data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(), shotnum, args.dataset.lower(), i)
                ).type(torch.long).squeeze().cuda()

            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9

            for _ in range(800):
                opt.zero_grad()
                if args.downstream_task == 'graph':
                    logits = log(features, adj_t, sparse, model.gcn, idx_train, batch_train, lbls_train, 1,
                                 raw_adj=adj_raw_t, step=global_step).float().cuda()
                else:
                    logits = log(features, adj_t, sparse, model.gcn, idx_train, lbls_train, 1,
                                 raw_adj=adj_raw_t, step=global_step).float().cuda()

                loss = xent(logits, lbls_train)

                if args.entropy_weight > 0 and hasattr(log.downstreamPrompt, 'last_lambda_list'):
                    ent = 0.0
                    for lam in log.downstreamPrompt.last_lambda_list:
                        ent = ent + (-lam * torch.log(lam + 1e-9)).sum(dim=-1).mean()
                    loss = loss + args.entropy_weight * ent

                if loss < best:
                    best = loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    break

                loss.backward()
                opt.step()
                global_step += 1

            if args.downstream_task == 'graph':
                logits = log(features, adj_t, sparse, model.gcn, test_index, test_batch, raw_adj=adj_raw_t)
            else:
                logits = log(features, adj_t, sparse, model.gcn, idx_test, raw_adj=adj_raw_t)

            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')

            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)
            tot += acc

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
        logging.info('Mean:[{:.4f}]'.format(acc_mean))
        logging.info('Std :[{:.4f}]'.format(acc_std))

        with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, dialect="excel")
            writer.writerow([
                pretrain_dataset_str,
                downstream_dataset,
                f"{acc_mean:.3f}", f"{acc_std:.3f}",
                f"{microf_mean:.3f}", f"{microf_std:.3f}",
                f"{macrof_mean:.3f}", f"{macrof_std:.3f}",
            ])