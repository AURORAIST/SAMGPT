import csv
import logging
import os
import copy

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from downprompt import downprompt
from preprompt import pca_compression
from utils import process
from utils.dataset import load_dataset


# --- PyTorch 2.6+ compatibility: torch.load defaults weights_only=True ---
# The cache files in this project may contain scipy.sparse objects, so we
# explicitly allow trusted local deserialization.
def torch_load_trusted(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


# Convenience helper for state_dict checkpoints.
def load_state_dict_trusted(path: str, map_location=None):
    sd = torch_load_trusted(path, map_location=map_location)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict) in {path}, got {type(sd)}")
    return sd


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


def prepare_downstream_eval_context(args, parent_directory, device, unify_dim, sparse):
    data_root = os.path.join(parent_directory, 'data')
    downstream_dataset = load_dataset(args.dataset, path=data_root)
    downstream_loader = DataLoader(downstream_dataset)
    downstream_context = {}
    for data in downstream_loader:
        features, adj = process.process_tu(data, data.x.shape[1])
        features = torch.FloatTensor(pca_compression(features, k=unify_dim)).to(device)

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_sp = adj

        idx_test = range(int(data.y.shape[0] - 100), data.y.shape[0])
        labels = data.y
        nb_classes = len(np.unique(np.array(data.y)))

        if args.downstream_task == 'graph':
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
            test_index = None
            test_batch = None

        if sparse:
            adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        else:
            adj = torch.FloatTensor(adj.todense()).to(device)

        downstream_context = {
            'features': features,
            'adj': adj,
            'adj_sp': adj_sp,
            'labels': labels,
            'idx_test': idx_test,
            'nb_classes': nb_classes,
            'downstream_dataset': downstream_dataset,
            'test_index': test_index,
            'test_batch': test_batch,
        }

    return downstream_context


def run_downstream_validation(
    args,
    model,
    downstream_context,
    device,
    parent_directory,
    pretrain_dataset_str,
    shot_num,
    num_layers_num,
    hid_units,
    unify_dim,
    sparse,
    csv_name,
    stage_label='final',
    write_csv=True,
):
    features = downstream_context['features']
    adj = downstream_context['adj']
    adj_sp = downstream_context['adj_sp']
    labels = downstream_context['labels']
    idx_test = downstream_context['idx_test']
    nb_classes = downstream_context['nb_classes']
    downstream_dataset = downstream_context['downstream_dataset']
    test_index = downstream_context['test_index']
    test_batch = downstream_context['test_batch']

    eval_start_msg = (
        f'[EVAL] start downstream_task={args.downstream_task} '
        f'query_mode={args.query_mode} subgraph_hop={args.subgraph_hop} '
        f'max_neighbors={args.max_neighbors} readout={args.readout} prototype={args.prototype} '
        f'stage={stage_label}'
    )
    print(eval_start_msg)
    logging.info(eval_start_msg)

    model_params = list(model.parameters())
    original_requires_grad = [param.requires_grad for param in model_params]
    original_training = model.training
    for param in model_params:
        param.requires_grad_(False)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    try:
        with torch.no_grad():
            embeds, _ = model.embed(features, adj, sparse, None, args.pretrain_method == 'LP')

        downstreamlrlist = [0.001]

        for downstreamlr in downstreamlrlist:
            test_lbls = labels[idx_test].to(device)
            print('-' * 100)

            for shotnum in range(1, shot_num + 1):
                tot = torch.zeros(1).to(device)
                accs = []
                macrof = []
                microf = []
                print('shotnum', shotnum)

                split_ids = range(args.split_start, args.split_start + args.num_splits)
                for i in tqdm(split_ids):
                    split_log_msg = (
                        f'[EVAL] split={i} shot={shotnum} '
                        f'downstream_task={args.downstream_task} query_mode={args.query_mode} '
                        f'subgraph_hop={args.subgraph_hop} max_neighbors={args.max_neighbors} '
                        f'stage={stage_label}'
                    )
                    print(split_log_msg)
                    logging.info(split_log_msg)

                    fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
                    combines.append(args.beta)

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

                        loss = criterion(logits, lbls_train)
                        if loss < best:
                            best = float(loss.detach())
                            cnt_wait = 0
                            best_state = copy.deepcopy(log.state_dict())
                        else:
                            cnt_wait += 1
                        if cnt_wait == 50:
                            break

                        loss.backward()
                        opt.step()

                    if best_state is not None:
                        log.load_state_dict(best_state)

                    if args.downstream_task == 'graph':
                        logits = log(features, adj, sparse, model.gcn, test_index, test_batch)
                    else:
                        logits = log(features, adj, sparse, model.gcn, idx_test, None, 0, args, adj_sp, sparse)

                    preds = torch.argmax(logits, dim=1).to(device)
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

                if write_csv:
                    with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
                        writer = csv.writer(file, dialect='excel')
                        writer.writerow(
                            [
                                pretrain_dataset_str,
                                downstream_dataset,
                                f"{acc_mean:.3f}",
                                f"{acc_std:.3f}",
                                f"{microf_mean:.3f}",
                                f"{microf_std:.3f}",
                                f"{macrof_mean:.3f}",
                                f"{macrof_std:.3f}",
                            ]
                        )

        return {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'microf_mean': microf_mean,
            'microf_std': microf_std,
            'macrof_mean': macrof_mean,
            'macrof_std': macrof_std,
        }
    finally:
        for param, requires_grad in zip(model_params, original_requires_grad):
            param.requires_grad_(requires_grad)
        model.train(original_training)
