import csv
import json
import logging
import os
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from preprompt import PrePrompt, pca_compression
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from srcgraphprompt.utils import process, usp_shared as _usp_shared
from srcgraphprompt.utils.dataset import load_dataset

_usp_device = _usp_shared._usp_device
_usp_read_fewshot = _usp_shared._usp_read_fewshot
_usp_load_graph = _usp_shared._usp_load_graph
_permute_target_edges = _usp_shared._permute_target_edges
_identity_like_adj = _usp_shared._identity_like_adj
_write_sanity_row = _usp_shared._write_sanity_row
_validate_neg_exclude_scope = _usp_shared._validate_neg_exclude_scope
_usp_feature_dropout = _usp_shared._usp_feature_dropout
_usp_edge_dropout = _usp_shared._usp_edge_dropout
_offset_ego_exclusion_rows = _usp_shared._offset_ego_exclusion_rows
_csr_mask_to_row_lists = _usp_shared._csr_mask_to_row_lists
_write_rows_csv = _usp_shared._write_rows_csv
_write_sanity_summary = _usp_shared._write_sanity_summary
_snapshot_named_params = _usp_shared._snapshot_named_params
_delta_norm_from_snapshot = _usp_shared._delta_norm_from_snapshot

def _run_sanity_calibration_legacy(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        build_subgraph_readout,
        domain_balanced_negative_indices,
        ego_subgraph_mask,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
        structural_alignment_loss,
        structural_role_buckets,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_sanity_calibration_v2.csv')
    if args.overwrite_sanity_table and os.path.exists(table_name):
        os.remove(table_name)
    sanity_rows = []
    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    aug_type = args.aug_type
    shot = args.shots[0]
    support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, args.seed, device)
    feature, _, _, target_adj = _usp_load_graph(
        target_domain,
        args.unify_dim,
        cache_dir,
        device,
        sparse=sparse,
        data_root=os.path.join(parent_directory, 'data'),
    )
    target_data = load_dataset(target_domain, path=os.path.join(parent_directory, 'data'))[0]
    labels = target_data.y.to(device).long()
    base_query_idx = torch.arange(int(labels.shape[0] - 100), labels.shape[0], device=device)
    query_idx = base_query_idx
    if torch.isin(support_idx, query_idx).any():
        raise ValueError("Support/query split overlap detected.")
    shuffled_labels_by_seed = {}
    for shuffle_seed in args.label_shuffle_seeds:
        gen = torch.Generator()
        gen.manual_seed(shuffle_seed)
        shuffled_labels = labels.clone()
        perm = torch.randperm(support_labels.numel(), generator=gen).to(device)
        shuffled_labels[support_idx] = support_labels[perm]
        shuffled_labels_by_seed[shuffle_seed] = shuffled_labels
    target_adj_edge_permuted = _permute_target_edges(target_adj, args.seed)

    def emit(
        method,
        internal_path,
        epoch,
        eval_type,
        metrics,
        train_time=0.0,
        loss_stats=None,
        label_shuffle=False,
        feature_permuted=False,
        edge_permuted=False,
        label_shuffle_seed=None,
    ):
        loss_stats = loss_stats or {}
        row = {
            'method': method,
            'internal_pretrain_path': internal_path,
            'epoch': epoch,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': args.seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'eval_type': eval_type,
            'label_shuffle': label_shuffle,
            'label_shuffle_seed': label_shuffle_seed,
            'feature_permuted': feature_permuted,
            'edge_permuted': edge_permuted,
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'acc': metrics['acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        sanity_rows.append(row)
        _write_sanity_row(table_name, row)
        print('SANITY result:', json.dumps(row, ensure_ascii=False))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # FeatureOnly + Prototype: projected input features, no GNN.
    feature_metrics = evaluate_fewshot_prototype(
        feature,
        feature,
        labels,
        support_idx,
        query_idx,
        query_mode='node',
        prototype_type=args.prototype,
        metric='cosine',
    )
    emit('FeatureOnlyPrototype', 'none', 0, 'SharedPrototypeEval', feature_metrics)
    for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
        feature_shuffle_metrics = evaluate_fewshot_prototype(
            feature,
            feature,
            shuffled_labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'FeatureOnlyPrototype_LabelShuffle',
            'none',
            0,
            'SharedPrototypeEval',
            feature_shuffle_metrics,
            label_shuffle=True,
            label_shuffle_seed=shuffle_seed,
        )
    perm = torch.randperm(feature.size(0), device=device)
    feature_permuted = feature[perm]
    feature_perm_metrics = evaluate_fewshot_prototype(
        feature_permuted,
        feature_permuted,
        labels,
        support_idx,
        query_idx,
        query_mode='node',
        prototype_type=args.prototype,
        metric='cosine',
    )
    emit('FeatureOnlyPrototype_FeaturePermuted', 'none', 0, 'SharedPrototypeEval', feature_perm_metrics, feature_permuted=True)
    emit('FeatureOnlyPrototype_EdgePermuted', 'none', 0, 'SharedPrototypeEval', feature_metrics, edge_permuted=True)

    # RandomEncoder + Prototype and NoPretrainGCN + Prototype are intentionally untrained.
    for method in ['RandomEncoderPrototype', 'NoPretrainGCNPrototype']:
        model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            max(1, len(source_domains)),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        model.eval()
        with torch.no_grad():
            h = model.gcn(feature, target_adj, sparse, False).squeeze(0)
            h_edge_permuted = model.gcn(feature, target_adj_edge_permuted, sparse, False).squeeze(0)
        metrics = evaluate_fewshot_prototype(
            h,
            h,
            labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(method, 'none', 0, 'SharedPrototypeEval', metrics)
        if method == 'NoPretrainGCNPrototype':
            for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
                shuffle_metrics = evaluate_fewshot_prototype(
                    h,
                    h,
                    shuffled_labels,
                    support_idx,
                    query_idx,
                    query_mode='node',
                    prototype_type=args.prototype,
                    metric='cosine',
                )
                emit(
                    'NoPretrainGCNPrototype_LabelShuffle',
                    'none',
                    0,
                    'SharedPrototypeEval',
                    shuffle_metrics,
                    label_shuffle=True,
                    label_shuffle_seed=shuffle_seed,
                )
            permuted_h = model.gcn(feature_permuted, target_adj, sparse, False).squeeze(0)
            perm_metrics = evaluate_fewshot_prototype(
                permuted_h,
                permuted_h,
                labels,
                support_idx,
                query_idx,
                query_mode='node',
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit('NoPretrainGCNPrototype_FeaturePermuted', 'none', 0, 'SharedPrototypeEval', perm_metrics, feature_permuted=True)
            edge_metrics = evaluate_fewshot_prototype(
                h_edge_permuted,
                h_edge_permuted,
                labels,
                support_idx,
                query_idx,
                query_mode='node',
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit('NoPretrainGCNPrototype_EdgePermuted', 'none', 0, 'SharedPrototypeEval', edge_metrics, edge_permuted=True)

    source_graphs = []
    for name in source_domains:
        src_feature, _, _, src_adj = _usp_load_graph(
            name,
            args.unify_dim,
            cache_dir,
            device,
            sparse=sparse,
            data_root=os.path.join(parent_directory, 'data'),
        )
        source_graphs.append((name, src_feature, src_adj))

    _validate_neg_exclude_scope(args, source_graphs)

    for epoch_count in args.sanity_epochs:
        # SAMGPT encoder + shared prototype evaluator. This uses GRAPHCL internally.
        start = time.time()
        sam_model = PrePrompt(
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
        opt = torch.optim.Adam(sam_model.parameters(), lr=args.lr, weight_decay=0.0)
        graphcl_losses = []
        for _ in range(epoch_count):
            opt.zero_grad()
            total = torch.tensor(0.0, device=device)
            losses = []
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                seq = torch.stack([src_feature, src_feature[torch.randperm(src_feature.size(0))], src_feature.detach(), src_feature.detach()])
                aug1 = _usp_edge_dropout(src_adj, args.drop_percent)
                aug2 = _usp_edge_dropout(src_adj, args.drop_percent)
                adjs = torch.stack([src_adj.to_dense(), aug1.to_dense(), aug2.to_dense()])
                lbl = torch.cat([torch.ones(1, src_feature.size(0)), torch.zeros(1, src_feature.size(0))], dim=1).to(device)
                logits = next(
                    sam_model.compute_prelogits_GRAPHCL(
                        [sam_model.feature_prompt_layers[graph_id]],
                        [sam_model.structure_prompt_layers[graph_id]],
                        [seq],
                        [adjs],
                        sparse=False,
                        msk=None,
                        samp_bias1=None,
                        samp_bias2=None,
                        aug_type=aug_type,
                    )
                )
                loss = sam_model.loss(logits, lbl)
                total = total + loss
                losses.append(float(loss.detach()))
            total.backward()
            opt.step()
            graphcl_losses = losses
        sam_model.eval()
        with torch.no_grad():
            sam_h = sam_model.gcn(feature, target_adj, sparse, False).squeeze(0)
        sam_metrics = evaluate_fewshot_prototype(
            sam_h,
            sam_h,
            labels,
            support_idx,
            query_idx,
            query_mode='node',
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'SAMGPTPrototypeEval',
            'GRAPHCL',
            epoch_count,
            'SharedPrototypeEval',
            sam_metrics,
            train_time=time.time() - start,
            loss_stats={
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': None,
                'L_ss_final_epoch_mean': float(np.mean(graphcl_losses)) if graphcl_losses else None,
                'L_align_final_epoch_mean': None,
            },
        )

        # USP full with shared prototype evaluator.
        start = time.time()
        usp_model = PrePrompt(
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
            align_type=args.align_type if args.use_structure_token else 'none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        opt = torch.optim.Adam(list(usp_model.parameters()) + list(usp_head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(epoch_count):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts, bucket_parts = [], [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = usp_model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = usp_model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = usp_head(
                    usp_model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + args.usp_lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
                bucket_parts.append(structural_role_buckets(src_adj, out['subgraph_mask'], bucket=args.structure_bucket))
                if args.negative_sampling == 'domain_balanced' and args.neg_exclude_scope == 'ego':
                    ego_exclusion_rows.extend(_offset_ego_exclusion_rows(
                        _csr_mask_to_row_lists(out['subgraph_mask'].bool()),
                        node_offset,
                        device,
                    ))
                node_offset += out['node_embeds'].size(0)
            all_h = torch.cat(h_parts, dim=0)
            all_g = torch.cat(g_parts, dim=0)
            domain_ids = torch.cat(domain_parts, dim=0)
            role_buckets = torch.cat(bucket_parts, dim=0)
            if args.negative_sampling == 'domain_balanced':
                neg_idx, neg_stats = domain_balanced_negative_indices(
                    domain_ids,
                    num_negatives=args.num_negatives,
                    intra_neg_ratio=args.intra_neg_ratio,
                    ego_exclusion_rows=ego_exclusion_rows if ego_exclusion_rows else None,
                    exclude_scope=args.neg_exclude_scope,
                )
            else:
                neg_idx, neg_stats = random_negative_indices(all_h.size(0), args.num_negatives, device=device)
            loss_ns, _ = sampled_info_nce_indexed(
                all_h, all_g, all_g, neg_idx, temperature=args.usp_temperature
            )
            align_loss = torch.tensor(0.0, device=device)
            if args.align_type == 'placeholder_structural_role':
                align_loss = structural_alignment_loss(all_g, domain_ids, role_buckets, temperature=args.usp_temperature)
            total_loss = total_loss + args.usp_lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            total_loss.backward()
            opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }
        usp_model.eval()
        usp_head.eval()
        with torch.no_grad():
            target_h = usp_head.encode(usp_model.gcn, feature, target_adj, sparse=sparse)
            target_g, _ = usp_head.subgraph_embeddings(target_h, target_adj, k=args.subgraph_hop)
            target_h_edge_permuted = usp_head.encode(usp_model.gcn, feature, target_adj_edge_permuted, sparse=sparse)
            target_g_edge_permuted, _ = usp_head.subgraph_embeddings(
                target_h_edge_permuted,
                target_adj_edge_permuted,
                k=args.subgraph_hop,
            )
        usp_metrics = evaluate_fewshot_prototype(
            target_h,
            target_g,
            labels,
            support_idx,
            query_idx,
            query_mode=args.query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit('USPFull', 'USP', epoch_count, 'SharedPrototypeEval', usp_metrics, time.time() - start, loss_stats)
        for shuffle_seed, shuffled_labels in shuffled_labels_by_seed.items():
            usp_shuffle_metrics = evaluate_fewshot_prototype(
                target_h,
                target_g,
                shuffled_labels,
                support_idx,
                query_idx,
                query_mode=args.query_mode,
                prototype_type=args.prototype,
                metric='cosine',
            )
            emit(
                'USPFull_LabelShuffle',
                'USP',
                epoch_count,
                'SharedPrototypeEval',
                usp_shuffle_metrics,
                time.time() - start,
                loss_stats,
                label_shuffle=True,
                label_shuffle_seed=shuffle_seed,
            )
        usp_edge_metrics = evaluate_fewshot_prototype(
            target_h_edge_permuted,
            target_g_edge_permuted,
            labels,
            support_idx,
            query_idx,
            query_mode=args.query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(
            'USPFull_EdgePermuted',
            'USP',
            epoch_count,
            'SharedPrototypeEval',
            usp_edge_metrics,
            time.time() - start,
            loss_stats,
            edge_permuted=True,
        )

    if sanity_rows:
        tmp_table_name = table_name + '.tmp'
        with open(tmp_table_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sanity_rows)
        os.replace(tmp_table_name, table_name)

    return table_name


