import csv
import copy
import json
import logging
import os
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from downprompt import downprompt
from preprompt import PrePrompt, pca_compression
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from srcgraphprompt.utils import aug, process
from srcgraphprompt.utils.dataset import load_dataset
from srcgraphprompt.utils.usp_shared import *

def _run_usp_minimal_legacy(args, parent_directory, current_dir, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        ClassPrototypeSubgraphClassifier,
        domain_balanced_negative_indices,
        ego_subgraph_mask_block,
        ego_subgraph_mask,
        prototype_compactness,
        prototype_separation,
        random_negative_indices,
        sampled_info_nce_indexed,
        similarity_margin,
        structural_alignment_loss,
        structural_role_buckets,
    )

    start_time = time.time()

    def _emit_eval_log(message):
        print(message)
        logging.info(message)

    device = _usp_device(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    save_dir = os.path.join(parent_directory, 'checkpoints')
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    source_domains = args.pretrain_datasets
    source_tag = '_'.join(source_domains)
    target_domain = args.dataset
    use_structure_token = args.use_structure_token
    set_name = (
        f'usp_{source_tag}_to_{target_domain}_k{args.subgraph_hop}_'
        f'mn{args.max_neighbors}_{args.readout}_{args.query_mode}_{args.prototype}_seed{args.seed}'
    )
    save_name = os.path.join(save_dir, f'{set_name}.pkl')
    jsonl_name = os.path.join(result_dir, 'usp_minimal_runs.jsonl')
    table_name = os.path.join(result_dir, 'usp_minimal_table.csv')
    time_profile_path = os.path.join(result_dir, 'usp_time_profile.csv')
    time_profile_live_path = os.path.join(result_dir, 'usp_time_profile_live.csv')
    if args.usp_eval_protocol == 'samgpt_standard':
        table_name = os.path.join(result_dir, 'usp_samgpt_standard_eval.csv')

    def _append_row_csv(path, row):
        fieldnames = list(row.keys())
        if not os.path.exists(path):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        merged_fieldnames = list(existing_fieldnames)
        for name in fieldnames:
            if name not in merged_fieldnames:
                merged_fieldnames.append(name)
        if merged_fieldnames == existing_fieldnames:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writerow(row)
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)

    source_graphs = []
    if args.usp_epochs > 0:
        for name in source_domains:
            feature, _, _, torch_adj = _usp_load_graph(
                name,
                args.unify_dim,
                cache_dir,
                device,
                sparse=sparse,
                data_root=os.path.join(parent_directory, 'data'),
            )
            source_graphs.append((name, feature, torch_adj))
    else:
        source_graphs = [(name, None, None) for name in source_domains]

    cached_masks = {}
    cache_threshold = 50_000_000
    if args.usp_epochs > 0:
        for name, _, adj in tqdm(source_graphs, desc='Caching ego-subgraph masks'):
            cached_masks[name] = build_or_load_ego_cache(
                name=name,
                adj=adj,
                cache_dir=cache_dir,
                result_dir=result_dir,
                k=args.subgraph_hop,
                max_neighbors=args.max_neighbors,
                cache_threshold=cache_threshold,
                block_size=512,
            )
        _validate_neg_exclude_scope(args, source_graphs, cache_threshold=cache_threshold)

    model = PrePrompt(
        args.unify_dim,
        args.hid_units,
        'prelu',
        len(source_graphs),
        args.layers_num,
        0.1,
        type_=args.combinetype,
        backbone=args.backbone,
        alpha=args.alpha,
        ablation=args.ablation_pre,
    ).to(device)
    usp_head = USPPretrainingHead(
        args.hid_units,
        readout=args.readout,
        k=args.subgraph_hop,
        temperature=args.usp_temperature,
        lambda_ss=args.usp_lambda_ss,
        lambda_align=args.usp_lambda_align,
        lambda_reg=0.0,
        align_type=args.align_type if args.use_structure_token else 'none',
        structure_bucket=args.structure_bucket,
        max_neighbors=args.max_neighbors,
        use_csr_readout=args.csr_readout,
        ss_loss_mode=args.ss_loss_mode,
        ss_num_negatives=args.ss_num_negatives,
        aug_mask_policy=args.aug_mask_policy,
    ).to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(usp_head.parameters()), lr=args.lr, weight_decay=0.0)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=amp_enabled)
    last_stats = {
        'L_ns_final_epoch_mean': None,
        'L_ss_final_epoch_mean': None,
        'L_align_final_epoch_mean': None,
        'loss_stat_type': None,
        'num_intra_neg': None,
        'num_inter_neg': None,
        'num_negatives_requested': args.num_negatives,
        'num_negatives_effective': None,
        'intra_neg_ratio_requested': args.intra_neg_ratio,
        'intra_neg_ratio_effective': None,
        'negative_fallback_reason': None,
        'negative_sampling_effective': None,
    }
    best_loss = float('inf')
    total_nodes = sum(feature.size(0) for _, feature, _ in source_graphs if feature is not None)
    cached_neg = None
    neg_refresh_count = 0
    neg_reuse_count = 0
    cached_neg_by_batch = {}  # 跨 epoch 共享的缓存，使得 neg_refresh_interval 能正常工作
    downstream_eval_interval = getattr(args, 'downstream_eval_interval', 50)
    upstream_loss_history = []
    upstream_lns_history = []
    upstream_lss_history = []

    target_feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device)
    nb_classes = int(labels.max().item() + 1)
    idx_test = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    test_lbls = labels[idx_test].type(torch.long)
    target_adj_cpu_dense = target_adj.to_dense().detach().cpu() if target_adj.is_sparse else target_adj.detach().cpu()

    def _run_downstream_eval(stage_label='final', upstream_epoch=None):
        model.eval()
        usp_head.eval()
        use_downprompt_head = getattr(args, 'usp_down_head', 'prototype') == 'downprompt'
        node_embeds = None
        subgraph_embeds = None
        if not use_downprompt_head:
            with torch.no_grad():
                node_embeds = usp_head.encode(model.gcn, target_feature, target_adj, sparse=sparse, prompt_layers=None)
                subgraph_embeds, _ = usp_head.subgraph_embeddings(node_embeds, target_adj, k=args.subgraph_hop)

        representation_source = {
            'node': 'node_embedding',
            'subgraph': 'subgraph_readout',
            'hybrid': 'hybrid(node+subgraph)',
        }.get(args.query_mode, f'unknown({args.query_mode})')
        _emit_eval_log(
            f"[EVAL] start downstream_task={args.downstream_task} query_mode={args.query_mode} "
            f"subgraph_hop={args.subgraph_hop} max_neighbors_per_node={args.max_neighbors} "
            f"readout={args.readout} prototype={args.prototype} down_head={getattr(args, 'usp_down_head', 'prototype')} stage={stage_label}"
        )
        _emit_eval_log(f"[EVAL] representation_source={representation_source}")

        def _subgraph_size_stats(node_idx_tensor):
            if node_idx_tensor is None or node_idx_tensor.numel() == 0:
                return 0.0, 0
            mask_block = ego_subgraph_mask_block(
                target_adj_cpu_dense,
                node_idx_tensor.detach().cpu(),
                k=args.subgraph_hop,
                max_neighbors=args.max_neighbors,
            )
            sizes = mask_block.sum(dim=1).float()
            return float(sizes.mean().item()), int(sizes.max().item())

        def _safe_plot_token(value):
            return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))

        def evaluate_one_split(shot, split):
            support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, split, device)
            support_set = set(support_idx.detach().cpu().tolist())
            query_set = set(idx_test.detach().cpu().tolist())
            support_query_overlap = len(support_set.intersection(query_set))
            support_label_match = torch.equal(labels[support_idx].long(), support_labels.long())
            if not support_label_match:
                raise ValueError(f"Few-shot support labels mismatch at shot={shot}, split={split}.")
            _emit_eval_log(
                f"[EVAL] split={split} shot={shot} downstream_task={args.downstream_task} "
                f"query_mode={args.query_mode} subgraph_hop={args.subgraph_hop} "
                f"max_neighbors_per_node={args.max_neighbors} stage={stage_label}"
            )
            _emit_eval_log(
                f"[EVAL][split_integrity] split={split} shot={shot} "
                f"support_query_overlap={support_query_overlap} support_label_match={support_label_match}"
            )

            if use_downprompt_head:
                model_params = list(model.parameters())
                model_requires_grad = [param.requires_grad for param in model_params]
                for param in model_params:
                    param.requires_grad_(False)
                try:
                    fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
                    combines = list(combines)
                    combines.append(args.beta)
                    classifier = downprompt(
                        args.hid_units,
                        nb_classes,
                        args.unify_dim,
                        args.layers_num,
                        fea_pretext_weights,
                        str_pretext_weights,
                        combines,
                        args.combinetype,
                        args.ablation_down,
                    ).to(device)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
                    best_loss = float('inf')
                    best_state = None
                    loss_history = []
                    for epoch in range(args.usp_down_epochs):
                        classifier.train()
                        optimizer.zero_grad()
                        logits = classifier(
                            target_feature,
                            target_adj,
                            sparse,
                            model.gcn,
                            support_idx,
                            support_labels,
                            1,
                            args,
                            target_adj_cpu_dense,
                            False,
                        )
                        loss = F.cross_entropy(logits, support_labels)
                        loss_history.append(float(loss.detach()))
                        loss.backward()
                        optimizer.step()
                        if float(loss.detach()) < best_loss:
                            best_loss = float(loss.detach())
                            best_state = copy.deepcopy(classifier.state_dict())
                        should_log = ((epoch + 1) % max(int(getattr(args, 'usp_down_log_interval', 10)), 1) == 0) or (epoch + 1 == args.usp_down_epochs)
                        if should_log:
                            with torch.no_grad():
                                support_acc = (torch.argmax(logits, dim=1) == support_labels).float().mean().item() * 100.0
                                eval_logits = classifier(
                                    target_feature,
                                    target_adj,
                                    sparse,
                                    model.gcn,
                                    idx_test,
                                    None,
                                    0,
                                    args,
                                    target_adj_cpu_dense,
                                    False,
                                )
                                eval_acc = (torch.argmax(eval_logits, dim=1) == test_lbls).float().mean().item() * 100.0
                            msg = (
                                f'[EVAL][{stage_label}][shot={shot}][split={split}] '
                                f'epoch {epoch + 1}/{args.usp_down_epochs} loss={loss.item():.6f} '
                                f'support_acc={support_acc:.2f} eval_acc={eval_acc:.2f}'
                            )
                            _emit_eval_log(msg)
                    if best_state is not None:
                        classifier.load_state_dict(best_state)

                    classifier.eval()
                    with torch.no_grad():
                        classifier(
                            target_feature,
                            target_adj,
                            sparse,
                            model.gcn,
                            support_idx,
                            support_labels,
                            1,
                            args,
                            target_adj_cpu_dense,
                            False,
                        )
                        logits = classifier(
                            target_feature,
                            target_adj,
                            sparse,
                            model.gcn,
                            idx_test,
                            None,
                            0,
                            args,
                            target_adj_cpu_dense,
                            False,
                        )
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == test_lbls).float().mean().item() * 100.0
                        preds_cpu = preds.cpu().numpy()
                        test_cpu = test_lbls.cpu().numpy()
                        micro_f1 = f1_score(test_cpu, preds_cpu, average='micro') * 100.0
                        macro_f1 = f1_score(test_cpu, preds_cpu, average='macro') * 100.0
                        margin = similarity_margin(logits, test_lbls).mean().item()
                        support_repr = classifier.make_query(
                            target_feature,
                            target_adj,
                            sparse,
                            model.gcn,
                            support_idx,
                            args=args,
                            adj_for_subgraph=target_adj_cpu_dense,
                            adj_for_subgraph_is_sparse=False,
                        )
                        compactness = prototype_compactness(support_repr, support_labels, classifier.ave.to(support_repr.device)).item()
                        separation = prototype_separation(classifier.ave.to(support_repr.device)).item()
                    return {
                        'acc': acc,
                        'macro_f1': macro_f1,
                        'micro_f1': micro_f1,
                        'margin': margin,
                        'compactness': compactness,
                        'separation': separation,
                        'loss_history': loss_history,
                        'support_query_overlap': support_query_overlap,
                        'support_label_match': support_label_match,
                    }
                finally:
                    for param, requires_grad in zip(model_params, model_requires_grad):
                        param.requires_grad_(requires_grad)

            if args.query_mode == 'subgraph':
                avg_support_size, max_support_size = _subgraph_size_stats(support_idx)
                avg_query_size, max_query_size = _subgraph_size_stats(idx_test)
                _emit_eval_log(f"[EVAL] avg_support_subgraph_size={avg_support_size}")
                _emit_eval_log(f"[EVAL] avg_query_subgraph_size={avg_query_size}")
                _emit_eval_log(f"[EVAL] max_subgraph_size_effective={max(max_support_size, max_query_size)}")
                _emit_eval_log(f"[EVAL] support_repr_shape={tuple(subgraph_embeds[support_idx].shape)}")
                _emit_eval_log(f"[EVAL] query_repr_shape={tuple(subgraph_embeds[idx_test].shape)}")
            elif args.query_mode == 'node':
                _emit_eval_log(f"[EVAL] support_repr_shape={tuple(node_embeds[support_idx].shape)}")
                _emit_eval_log(f"[EVAL] query_repr_shape={tuple(node_embeds[idx_test].shape)}")
            else:
                _emit_eval_log(
                    f"[EVAL] support_repr_shape=node{tuple(node_embeds[support_idx].shape)}+subgraph{tuple(subgraph_embeds[support_idx].shape)}"
                )
                _emit_eval_log(
                    f"[EVAL] query_repr_shape=node{tuple(node_embeds[idx_test].shape)}+subgraph{tuple(subgraph_embeds[idx_test].shape)}"
                )

            classifier = ClassPrototypeSubgraphClassifier(
                args.hid_units,
                nb_classes,
                prototype=args.prototype,
                query=args.query_mode,
                temperature=args.usp_temperature,
                learnable_eta=True,
                eta=0.5,
            ).to(device)
            train_stats = _usp_train_downstream_classifier(
                classifier,
                node_embeds.detach(),
                subgraph_embeds.detach(),
                support_idx,
                support_labels,
                args.usp_down_epochs,
                args.lr,
                log_prefix=f'[EVAL][{stage_label}][shot={shot}][split={split}]',
                log_interval=getattr(args, 'usp_down_log_interval', 10),
                eval_idx=idx_test,
                eval_labels=test_lbls,
            )

            classifier.eval()
            with torch.no_grad():
                classifier(
                    node_embeds,
                    subgraph_embeds,
                    support_idx,
                    support_idx=support_idx,
                    support_labels=support_labels,
                    update_prototypes=True,
                )
                logits = classifier(node_embeds, subgraph_embeds, idx_test)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == test_lbls).float().mean().item() * 100.0
                preds_cpu = preds.cpu().numpy()
                test_cpu = test_lbls.cpu().numpy()
                micro_f1 = f1_score(test_cpu, preds_cpu, average='micro') * 100.0
                macro_f1 = f1_score(test_cpu, preds_cpu, average='macro') * 100.0
                margin = similarity_margin(logits, test_lbls).mean().item()
                support_repr = classifier.make_query(node_embeds[support_idx], subgraph_embeds[support_idx])
                compactness = prototype_compactness(support_repr, support_labels, classifier.class_prototypes).item()
                separation = prototype_separation(classifier.class_prototypes).item()
            return {
                'acc': acc,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'margin': margin,
                'compactness': compactness,
                'separation': separation,
                'loss_history': train_stats.get('loss_history', []) if train_stats else [],
                'support_query_overlap': support_query_overlap,
                'support_label_match': support_label_match,
            }

        for shot in args.shots:
            if args.usp_eval_protocol == 'samgpt_standard':
                split_ids = range(args.split_start, args.split_start + args.num_splits)
                split_metrics = []
                for split in tqdm(split_ids, desc=f'USP {stage_label} SAMGPT-standard {shot}-shot'):
                    metrics = evaluate_one_split(shot, split)
                    metrics['split'] = split
                    split_metrics.append(metrics)
                metric_keys = ['acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']
                means = {key: float(np.mean([m[key] for m in split_metrics])) for key in metric_keys}
                stds = {f'{key}_std': float(np.std([m[key] for m in split_metrics], ddof=1)) if len(split_metrics) > 1 else 0.0 for key in metric_keys}
                split = f'{args.split_start}-{args.split_start + args.num_splits - 1}'
            else:
                split = args.seed
                means = evaluate_one_split(shot, split)
                means['split'] = split
                split_metrics = [means]
                stds = {f'{key}_std': None for key in ['acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']}

            loss_curves = [
                {'label': f"split={m['split']}", 'values': m.get('loss_history', [])}
                for m in split_metrics
            ]
            support_query_overlap_total = int(sum(m.get('support_query_overlap', 0) for m in split_metrics))
            support_query_overlap_splits = int(sum(1 for m in split_metrics if m.get('support_query_overlap', 0) > 0))
            support_label_match_all = bool(all(m.get('support_label_match', False) for m in split_metrics))
            plot_name = (
                f"usp_downstream_loss_{_safe_plot_token(stage_label)}_"
                f"shot{shot}_splits{_safe_plot_token(split)}.png"
            )
            _usp_save_loss_curves_plot(
                loss_curves,
                os.path.join(result_dir, 'figures', plot_name),
                title=f'USP {stage_label} {shot}-shot downstream loss',
            )

            row = {
                'method': 'USP',
                'internal_pretrain_path': f'USP@{stage_label}',
                'setting': 'source_only',
                'eval_protocol': args.usp_eval_protocol,
                'source_domains': source_domains,
                'target_domain': target_domain,
                'shot': shot,
                'seed': args.seed,
                'split': split,
                'num_splits': args.num_splits if args.usp_eval_protocol == 'samgpt_standard' else 1,
                'test_nodes': int(idx_test.numel()),
                'support_query_overlap_total': support_query_overlap_total,
                'support_query_overlap_splits': support_query_overlap_splits,
                'support_label_match_all': support_label_match_all,
                'adaptation_setting': 'transductive',
                'target_unlabeled_used': True,
                'target_edges_used': True,
                'label_shuffle': False,
                'feature_permuted': False,
                'subgraph_hop': args.subgraph_hop,
                'readout': args.readout,
                'query_mode': args.query_mode,
                'down_head': getattr(args, 'usp_down_head', 'prototype'),
                'prototype': args.prototype,
                'negative_sampling': args.negative_sampling,
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
                'neg_refresh_count': neg_refresh_count,
                'neg_reuse_count': neg_reuse_count,
                'num_intra_neg': last_stats['num_intra_neg'],
                'num_inter_neg': last_stats['num_inter_neg'],
                'num_negatives_requested': last_stats['num_negatives_requested'],
                'num_negatives_effective': last_stats['num_negatives_effective'],
                'intra_neg_ratio_requested': last_stats['intra_neg_ratio_requested'],
                'intra_neg_ratio_effective': last_stats['intra_neg_ratio_effective'],
                'negative_fallback_reason': last_stats['negative_fallback_reason'],
                'negative_sampling_effective': last_stats['negative_sampling_effective'],
                'use_structure_token': use_structure_token,
                'align_type': args.align_type if args.use_structure_token else 'none',
                'L_ns_final_epoch_mean': last_stats['L_ns_final_epoch_mean'],
                'L_ss_final_epoch_mean': last_stats['L_ss_final_epoch_mean'],
                'L_align_final_epoch_mean': last_stats['L_align_final_epoch_mean'],
                'loss_stat_type': last_stats['loss_stat_type'],
                'train_time': time.time() - start_time,
                'acc': means['acc'],
                'acc_std': stds['acc_std'],
                'macro_f1': means['macro_f1'],
                'macro_f1_std': stds['macro_f1_std'],
                'micro_f1': means['micro_f1'],
                'micro_f1_std': stds['micro_f1_std'],
                'margin': means['margin'],
                'margin_std': stds['margin_std'],
                'compactness': means['compactness'],
                'compactness_std': stds['compactness_std'],
                'separation': means['separation'],
                'separation_std': stds['separation_std'],
            }
            with open(jsonl_name, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            _append_row_csv(table_name, row)
            print('USP result:', json.dumps(row, ensure_ascii=False))

    def _run_periodic_downstream_eval(stage_label, upstream_epoch):
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            _run_downstream_eval(stage_label=stage_label, upstream_epoch=upstream_epoch)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    for epoch in range(args.usp_epochs):
        epoch_start = time.time()
        forward_time = 0.0
        neg_sampling_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        graph_times = {}
        if args.progress_detail:
            tqdm.write(f"\n[USP] ===== start epoch {epoch + 1}/{args.usp_epochs} =====")
        model.train()
        usp_head.train()
        source_batch_size = args.source_batch_size if args.source_batch_size and args.source_batch_size > 0 else len(source_graphs)
        source_batch_size = max(1, min(source_batch_size, len(source_graphs)))
        epoch_loss_total = 0.0
        epoch_loss_ns_values = []
        epoch_loss_align_values = []
        ss_losses = []
        # neg_debug_enabled = os.environ.get('USP_NEG_DEBUG', '0') == '1'
        neg_debug_enabled = False  # DISABLED: debug output overhead

        # 每个 epoch 对 source_graphs 顺序进行随机打乱，使得跨域负采样更丰富
        # （使用 seed+epoch 保证可复现）
        epoch_order = list(range(len(source_graphs)))
        random.Random(args.seed + epoch).shuffle(epoch_order)

        for batch_start in range(0, len(epoch_order), source_batch_size):
            batch_end = min(batch_start + source_batch_size, len(epoch_order))
            batch_indices = epoch_order[batch_start:batch_end]
            batch_source_graphs = [
                (orig_graph_id, source_graphs[orig_graph_id])
                for orig_graph_id in batch_indices
            ]
            # 使用真实图名作为缓存 key；graph_id 保持为原始 source graph id
            batch_graph_ids = tuple(source_graphs[orig_graph_id][0] for orig_graph_id in batch_indices)
            batch_row_spans = []

            opt.zero_grad(set_to_none=True)
            batch_total_loss = torch.tensor(0.0, device=device)
            batch_h_parts = []
            batch_g_parts = []
            batch_domain_parts = []
            batch_bucket_parts = []
            batch_ego_exclusion_rows = []
            batch_ss_losses = []
            batch_node_offset = 0

            graph_iter = tqdm(
                batch_source_graphs,
                desc=f"USP epoch {epoch + 1}/{args.usp_epochs} graphs[{batch_start}:{batch_end}]",
                leave=False,
                dynamic_ncols=True,
                disable=True,
            )
            for graph_id, (name, feature, adj) in graph_iter:
                graph_t0 = time.time()
                cache_obj = cached_masks.get(name)
                if isinstance(cache_obj, dict):
                    cache_desc = cache_obj.get("cache_type", cache_obj.get("type", "dict"))
                elif cache_obj is None:
                    cache_desc = "none"
                else:
                    cache_desc = "full_tensor"
                graph_iter.set_postfix_str(f"{name}, N={feature.size(0)}, cache={cache_desc}")
                feature_in = feature
                prompt_layers = None
                prompt_elapsed = 0.0
                if use_structure_token:
                    t_prompt = time.time()
                    feature_in = model.feature_prompt_layers[graph_id](feature)
                    prompt_layers = model.structure_prompt_layers[graph_id]
                    prompt_elapsed = time.time() - t_prompt

                t_aug = time.time()
                aug_feature_1 = _usp_feature_dropout(feature_in, args.drop_percent)
                aug_feature_2 = _usp_feature_dropout(feature_in, args.drop_percent)
                aug_adj_1 = _usp_edge_dropout(adj, args.drop_percent)
                aug_adj_2 = _usp_edge_dropout(adj, args.drop_percent)
                aug_elapsed = time.time() - t_aug

                graph_start = time.time()
                out = usp_head(
                    model.gcn,
                    feature_in,
                    adj,
                    sparse=sparse,
                    aug_features_1=aug_feature_1,
                    aug_adj_1=aug_adj_1,
                    aug_features_2=aug_feature_2,
                    aug_adj_2=aug_adj_2,
                    prompt_layers=prompt_layers,
                    align_loss=None,
                    cached_mask=cached_masks.get(name),
                    compute_local_ns=args.compute_local_ns,
                    progress_blocks=args.progress_blocks,
                    progress_prefix=f"[USP][epoch {epoch + 1}][{name}]",
                )
                _sync_if_cuda(device)
                forward_elapsed = time.time() - graph_start
                forward_time += forward_elapsed
                graph_elapsed = time.time() - graph_t0
                graph_times[name] = forward_elapsed
                if args.progress_detail:
                    mem = _cuda_mem_mb(device)
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}] graph={graph_id + 1}/{len(source_graphs)} "
                        f"name={name} N={feature.size(0)} cache={cache_desc} "
                        f"prompt={prompt_elapsed:.2f}s aug={aug_elapsed:.2f}s "
                        f"forward={forward_elapsed:.2f}s total={graph_elapsed:.2f}s "
                        f"L_ss={float(out['loss_ss'].detach()):.6f}, "
                        f"gpu_alloc={mem['gpu_allocated_mb']}"
                    )
                batch_total_loss += args.usp_lambda_ss * out['loss_ss']
                batch_ss_losses.append(float(out['loss_ss'].detach()))
                ss_losses.append(float(out['loss_ss'].detach()))
                batch_h_parts.append(out['node_embeds'])
                batch_g_parts.append(out['subgraph_embeds'])
                batch_domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                batch_row_spans.append((graph_id, name, batch_node_offset, batch_node_offset + out['node_embeds'].size(0)))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    cache_rows = _build_ego_exclusion_rows_from_cache(cache_obj)
                    batch_ego_exclusion_rows.extend(_offset_ego_exclusion_rows(cache_rows, batch_node_offset, device))
                batch_node_offset += out['node_embeds'].size(0)
                if args.align_type == 'placeholder_structural_role':
                    batch_bucket_parts.append(structural_role_buckets(adj, out['subgraph_mask'], bucket=args.structure_bucket))

            concat_start = time.time()
            all_h = torch.cat(batch_h_parts, dim=0)
            all_g = torch.cat(batch_g_parts, dim=0)
            domain_ids = torch.cat(batch_domain_parts, dim=0)
            role_buckets = torch.cat(batch_bucket_parts, dim=0) if batch_bucket_parts else None
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] concat done, "
                    f"all_nodes={all_h.size(0)}, time={time.time() - concat_start:.2f}s"
                )
            neg_start = time.time()
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] negative sampling start: "
                    f"mode={args.negative_sampling}, scope={args.neg_exclude_scope}, "
                    f"refresh_interval={args.neg_refresh_interval}, "
                    f"N={all_h.size(0)}, K={args.num_negatives}"
                )
            if args.negative_sampling == 'domain_balanced':
                if batch_graph_ids not in cached_neg_by_batch or epoch == 0 or (epoch % args.neg_refresh_interval == 0):
                    neg_idx, neg_stats = domain_balanced_negative_indices(
                        domain_ids,
                        num_negatives=args.num_negatives,
                        intra_neg_ratio=args.intra_neg_ratio,
                        ego_exclusion_rows=batch_ego_exclusion_rows if batch_ego_exclusion_rows else None,
                        exclude_scope=args.neg_exclude_scope,
                    )
                    cached_neg_by_batch[batch_graph_ids] = (neg_idx.detach().cpu(), neg_stats)
                    neg_refresh_count += 1
                    neg_event = 'refresh'
                else:
                    cached_neg_idx, neg_stats = cached_neg_by_batch[batch_graph_ids]
                    # 从缓存中复用时，将 CPU 缓存搬回当前设备，避免跨 batch/device 状态问题
                    neg_idx = cached_neg_idx.to(device)
                    neg_reuse_count += 1
                    neg_event = 'reuse'
                negative_sampling_effective = 'domain_balanced'
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
                negative_sampling_effective = 'random'
                neg_event = 'refresh'

            # DISABLED: Pubmed-specific negative sampling debug prints
            # if neg_debug_enabled and any(name == 'Pubmed' for _, name, _, _ in batch_row_spans):
            #     ... [debug code omitted] ...
            _sync_if_cuda(device)
            batch_neg_sampling_time = time.time() - neg_start
            neg_sampling_time += batch_neg_sampling_time
            if args.progress_detail:
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] negative sampling {neg_event} done, "
                    f"time={batch_neg_sampling_time:.2f}s, "
                    f"num_intra={neg_stats.get('num_intra_neg')}, "
                    f"num_inter={neg_stats.get('num_inter_neg')}, "
                    f"ego_enabled={neg_stats.get('ego_exclusion_enabled')}, "
                    f"avg_ego={neg_stats.get('avg_excluded_ego_size')}, "
                    f"fallback={neg_stats.get('fallback_reason') or neg_stats.get('negative_fallback_reason')}"
                )
            loss_start = time.time()
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                loss_ns, _ = sampled_info_nce_indexed(
                    all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
                )
                align_loss = torch.tensor(0.0, device=device)
                if args.align_type == 'placeholder_structural_role':
                    align_loss = structural_alignment_loss(all_g, domain_ids, role_buckets, temperature=args.usp_temperature)
                batch_total_loss = batch_total_loss + args.usp_lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            _sync_if_cuda(device)
            batch_loss_time = time.time() - loss_start
            loss_time += batch_loss_time
            epoch_loss_ns_values.append(float(loss_ns.detach()))
            epoch_loss_align_values.append(float(align_loss.detach()))
            if args.progress_detail:
                mean_ss = float(np.mean(batch_ss_losses)) if batch_ss_losses else 0.0
                tqdm.write(
                    f"[USP][epoch {epoch + 1}] loss done, "
                    f"time={batch_loss_time:.2f}s, "
                    f"L_ns={float(loss_ns.detach()):.6f}, "
                    f"L_ss={mean_ss:.6f}, "
                    f"L_align={float(align_loss.detach()):.6f}"
                )

            if torch.isfinite(batch_total_loss) and batch_total_loss.requires_grad:
                backward_start = time.time()
                scaler.scale(batch_total_loss).backward()
                scaler.step(opt)
                scaler.update()
                _sync_if_cuda(device)
                backward_time += time.time() - backward_start
                if args.progress_detail:
                    tqdm.write(
                        f"[USP][epoch {epoch + 1}] backward/step done, "
                        f"time={time.time() - backward_start:.2f}s"
                    )
            epoch_loss_total += float(batch_total_loss.detach())

        epoch_loss_mean = epoch_loss_total / max(1, len(source_graphs))

        if epoch_loss_mean < best_loss:
            best_loss = epoch_loss_mean
            torch.save({'model': model.state_dict(), 'usp_head': usp_head.state_dict()}, save_name)

        last_stats = {
            'L_ns_final_epoch_mean': float(np.mean(epoch_loss_ns_values)) if epoch_loss_ns_values else None,
            'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
            'L_align_final_epoch_mean': float(np.mean(epoch_loss_align_values)) if epoch_loss_align_values else None,
            'loss_stat_type': 'epoch_mean_over_source_graphs',
            'num_intra_neg': neg_stats.get('num_intra_neg'),
            'num_inter_neg': neg_stats.get('num_inter_neg'),
            'num_negatives_requested': args.num_negatives,
            'num_negatives_effective': neg_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': args.intra_neg_ratio,
            'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
            'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
            'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
            'fallback_reason': neg_stats.get('fallback_reason'),
            'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
            'negative_sampling_effective': negative_sampling_effective,
            'neg_exclude_scope': args.neg_exclude_scope,
            'neg_refresh_interval': args.neg_refresh_interval,
            'neg_refresh_count': neg_refresh_count,
            'neg_reuse_count': neg_reuse_count,
        }
        upstream_loss_history.append(float(epoch_loss_mean))
        upstream_lns_history.append(float(last_stats['L_ns_final_epoch_mean']))
        upstream_lss_history.append(float(last_stats['L_ss_final_epoch_mean']))
        print(
            f"USP epoch {epoch + 1}/{args.usp_epochs} "
            f"loss_sum={epoch_loss_total:.6f} "
            f"loss={epoch_loss_mean:.6f} "
            f"L_ns={last_stats['L_ns_final_epoch_mean']:.6f} "
            f"L_ss={last_stats['L_ss_final_epoch_mean']:.6f} "
            f"time={time.time() - epoch_start:.2f}s "
            f"neg={neg_sampling_time:.2f}s "
            f"backward={backward_time:.2f}s",
            flush=True,
        )
        if not np.isfinite(epoch_loss_mean):
            raise FloatingPointError('USP-SAM pretraining produced NaN/Inf loss.')

        if args.time_profile:
            mem = _cuda_mem_mb(device)
            _append_csv_row(
                time_profile_live_path,
                {
                    'epoch': epoch + 1,
                    'usp_epochs': args.usp_epochs,
                    'source_domains': '|'.join(source_domains),
                    'target_domain': target_domain,
                    'num_source_graphs': len(source_graphs),
                    'total_source_nodes': total_nodes,
                    'last_batch_nodes': int(all_h.size(0)),
                    'forward_time': forward_time,
                    'negative_sampling_time': neg_sampling_time,
                    'loss_time': loss_time,
                    'backward_time': backward_time,
                    'epoch_time': time.time() - epoch_start,
                    'negative_sampling': args.negative_sampling,
                    'neg_exclude_scope': args.neg_exclude_scope,
                    'neg_refresh_interval': args.neg_refresh_interval,
                    'neg_event': neg_event,
                    'neg_refresh_count': neg_refresh_count,
                    'neg_reuse_count': neg_reuse_count,
                    'num_negatives': args.num_negatives,
                    'intra_neg_ratio': args.intra_neg_ratio,
                    'L_ns': float(loss_ns.detach()),
                    'L_ss': float(np.mean(ss_losses)) if ss_losses else None,
                    'L_align': float(align_loss.detach()),
                    'loss_sum': float(epoch_loss_total),
                    'loss': float(epoch_loss_mean),
                    'graph_times_json': json.dumps(graph_times, ensure_ascii=False),
                    'gpu_allocated_mb': mem['gpu_allocated_mb'],
                    'gpu_reserved_mb': mem['gpu_reserved_mb'],
                },
            )

        if args.time_profile:
            if device.type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            else:
                mem_alloc = None
                mem_reserved = None
            _append_csv_row(
                time_profile_path,
                {
                    'epoch': epoch + 1,
                    'forward_time_sec': forward_time,
                    'neg_sampling_time_sec': neg_sampling_time,
                    'loss_time_sec': loss_time,
                    'backward_time_sec': backward_time,
                    'epoch_total_time_sec': time.time() - epoch_start,
                    'gpu_mem_allocated_mb': mem_alloc,
                    'gpu_mem_reserved_mb': mem_reserved,
                    'num_nodes_total': total_nodes,
                    'neg_refresh_count': neg_refresh_count,
                    'neg_reuse_count': neg_reuse_count,
                },
            )

        if downstream_eval_interval > 0 and (epoch + 1) % downstream_eval_interval == 0:
            _usp_save_loss_curves_plot(
                [
                    {'label': 'upstream_total_loss', 'values': upstream_loss_history},
                    {'label': 'upstream_L_ns', 'values': upstream_lns_history},
                    {'label': 'upstream_L_ss', 'values': upstream_lss_history},
                ],
                os.path.join(result_dir, 'figures', f'usp_upstream_loss_epoch_{epoch + 1}.png'),
                title=f'USP upstream loss through epoch {epoch + 1}',
                ylabel='Pretraining loss',
            )
            _emit_eval_log(f"[EVAL] periodic USP downstream validation at upstream epoch {epoch + 1}/{args.usp_epochs}")
            _run_periodic_downstream_eval(stage_label=f'upstream_epoch_{epoch + 1}', upstream_epoch=epoch + 1)

    if args.usp_epochs > 0 and os.path.exists(save_name):
        ckpt = _usp_safe_torch_load(save_name, map_location=device)
        model.load_state_dict(ckpt['model'])
        usp_head.load_state_dict(ckpt['usp_head'])
    if upstream_loss_history:
        _usp_save_loss_curves_plot(
            [
                {'label': 'upstream_total_loss', 'values': upstream_loss_history},
                {'label': 'upstream_L_ns', 'values': upstream_lns_history},
                {'label': 'upstream_L_ss', 'values': upstream_lss_history},
            ],
            os.path.join(result_dir, 'figures', 'usp_upstream_loss_final.png'),
            title='USP upstream loss final',
            ylabel='Pretraining loss',
        )
    final_stage = 'final' if args.usp_epochs > 0 else 'baseline_epoch0'
    _run_downstream_eval(stage_label=final_stage, upstream_epoch=args.usp_epochs if args.usp_epochs > 0 else 0)

    return jsonl_name


def _write_sanity_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
