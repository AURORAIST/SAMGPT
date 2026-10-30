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

from srcgraphprompt.utils import process
from srcgraphprompt.utils.dataset import load_dataset
from srcgraphprompt.utils.usp_shared import *

def _run_formal_batch1_legacy(args, parent_directory, sparse=True):
    from usp_sam import (
        USPPretrainingHead,
        domain_balanced_negative_indices,
        evaluate_fewshot_prototype,
        random_negative_indices,
        sampled_info_nce_indexed,
    )

    device = _usp_device(args)
    result_dir = os.path.join(parent_directory, 'result')
    cache_dir = os.path.join(parent_directory, 'cache')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    table_name = os.path.join(result_dir, 'usp_minimal_formal_batch1.csv')
    summary_name = os.path.join(result_dir, 'usp_minimal_formal_batch1_summary.csv')
    if args.overwrite_sanity_table:
        for path in [table_name, summary_name]:
            if os.path.exists(path):
                os.remove(path)

    source_domains = args.pretrain_datasets
    target_domain = args.dataset
    target_feature, _, _, target_adj = _usp_load_graph(
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

    rows = []

    def emit(
        method,
        internal_path,
        shot,
        seed,
        metrics,
        support_idx,
        support_labels,
        query_idx,
        train_time=0.0,
        loss_stats=None,
    ):
        loss_stats = loss_stats or {}
        split_stats = _split_integrity_stats(labels, support_idx, support_labels, query_idx, labels)
        row = {
            'method': method,
            'internal_pretrain_path': internal_path,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'shot': shot,
            'seed': seed,
            'adaptation_setting': 'transductive',
            'target_unlabeled_used': True,
            'target_edges_used': True,
            'subgraph_hop': args.subgraph_hop if method.startswith('USP') else None,
            'readout': args.readout if method.startswith('USP') else None,
            'query_mode': args.query_mode if method.startswith('USP') else 'node',
            'prototype': args.prototype,
            'negative_sampling': args.negative_sampling if method.startswith('USP') else None,
            'neg_exclude_scope': args.neg_exclude_scope if method.startswith('USP') else None,
            'neg_refresh_interval': args.neg_refresh_interval if method.startswith('USP') else None,
            'num_negatives_requested': loss_stats.get('num_negatives_requested'),
            'num_negatives_effective': loss_stats.get('num_negatives_effective'),
            'negative_fallback_reason': loss_stats.get('negative_fallback_reason'),
            'intra_neg_ratio_requested': loss_stats.get('intra_neg_ratio_requested'),
            'intra_neg_ratio_effective': loss_stats.get('intra_neg_ratio_effective'),
            'loss_stat_type': loss_stats.get('loss_stat_type'),
            'L_ns_final_epoch_mean': loss_stats.get('L_ns_final_epoch_mean'),
            'L_ss_final_epoch_mean': loss_stats.get('L_ss_final_epoch_mean'),
            'L_align_final_epoch_mean': loss_stats.get('L_align_final_epoch_mean'),
            'acc': metrics['acc'],
            'balanced_acc': metrics['balanced_acc'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'margin': metrics['margin'],
            'compactness': metrics['compactness'],
            'separation': metrics['separation'],
            'train_time': train_time,
            'device': str(device),
        }
        row.update(split_stats)
        rows.append(row)
        _write_sanity_row(table_name, row)
        print('FORMAL batch1 result:', json.dumps(row, ensure_ascii=False))

    def eval_embeddings(method, internal_path, node_embeds, subgraph_embeds, query_mode, shot, seed, train_time=0.0, loss_stats=None):
        support_idx, support_labels = _usp_read_fewshot(parent_directory, target_domain, shot, seed, device)
        query_idx = base_query_idx[~torch.isin(base_query_idx, support_idx)]
        metrics = evaluate_fewshot_prototype(
            node_embeds,
            subgraph_embeds,
            labels,
            support_idx,
            query_idx,
            query_mode=query_mode,
            prototype_type=args.prototype,
            metric='cosine',
        )
        emit(method, internal_path, shot, seed, metrics, support_idx, support_labels, query_idx, train_time, loss_stats)

    def train_samgpt(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        start = time.time()
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
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        graphcl_losses = []
        for _ in range(args.formal_epoch):
            opt.zero_grad()
            total = torch.tensor(0.0, device=device)
            losses = []
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                seq = torch.stack([
                    src_feature,
                    src_feature[torch.randperm(src_feature.size(0), device=device)],
                    src_feature.detach(),
                    src_feature.detach(),
                ])
                aug1 = _usp_edge_dropout(src_adj, args.drop_percent)
                aug2 = _usp_edge_dropout(src_adj, args.drop_percent)
                adjs = torch.stack([src_adj.to_dense(), aug1.to_dense(), aug2.to_dense()])
                lbl = torch.cat([torch.ones(1, src_feature.size(0)), torch.zeros(1, src_feature.size(0))], dim=1).to(device)
                logits = next(
                    model.compute_prelogits_GRAPHCL(
                        [model.feature_prompt_layers[graph_id]],
                        [model.structure_prompt_layers[graph_id]],
                        [seq],
                        [adjs],
                        sparse=False,
                        msk=None,
                        samp_bias1=None,
                        samp_bias2=None,
                        aug_type=aug_type,
                    )
                )
                loss = model.loss(logits, lbl)
                total = total + loss
                losses.append(float(loss.detach()))
            total.backward()
            opt.step()
            graphcl_losses = losses
        model.eval()
        with torch.no_grad():
            h = model.gcn(target_feature, target_adj, sparse, False).squeeze(0)
        return h, {
            'loss_stat_type': 'final_epoch_mean',
            'L_ns_final_epoch_mean': None,
            'L_ss_final_epoch_mean': float(np.mean(graphcl_losses)) if graphcl_losses else None,
            'L_align_final_epoch_mean': None,
        }, time.time() - start

    def train_usp(seed, method, lambda_ns, lambda_ss):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        start = time.time()
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
        head = USPPretrainingHead(
            args.hid_units,
            readout=args.readout,
            k=args.subgraph_hop,
            temperature=args.usp_temperature,
            align_type='none',
            structure_bucket=args.structure_bucket,
            use_csr_readout=args.csr_readout,
            ss_loss_mode=args.ss_loss_mode,
            ss_num_negatives=args.ss_num_negatives,
            aug_mask_policy=args.aug_mask_policy,
        ).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
        loss_stats = {}
        for _ in range(args.formal_epoch):
            opt.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            ss_losses = []
            h_parts, g_parts, mask_parts, domain_parts = [], [], [], []
            ego_exclusion_rows = []
            node_offset = 0
            for graph_id, (_, src_feature, src_adj) in enumerate(source_graphs):
                feature_in = model.feature_prompt_layers[graph_id](src_feature) if args.use_structure_token else src_feature
                prompt_layers = model.structure_prompt_layers[graph_id] if args.use_structure_token else None
                out = head(
                    model.gcn,
                    feature_in,
                    src_adj,
                    sparse=sparse,
                    aug_features_1=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_1=_usp_edge_dropout(src_adj, args.drop_percent),
                    aug_features_2=_usp_feature_dropout(feature_in, args.drop_percent),
                    aug_adj_2=_usp_edge_dropout(src_adj, args.drop_percent),
                    prompt_layers=prompt_layers,
                )
                total_loss = total_loss + lambda_ss * out['loss_ss']
                ss_losses.append(float(out['loss_ss'].detach()))
                h_parts.append(out['node_embeds'])
                g_parts.append(out['subgraph_embeds'])
                mask_parts.append(out['subgraph_mask'].bool())
                domain_parts.append(torch.full((out['node_embeds'].size(0),), graph_id, dtype=torch.long, device=device))
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
            total_loss = total_loss + lambda_ns * loss_ns + args.usp_lambda_align * align_loss
            if total_loss.requires_grad and float(total_loss.detach()) != 0.0:
                total_loss.backward()
                opt.step()
            loss_stats = {
                'loss_stat_type': 'final_epoch_mean',
                'L_ns_final_epoch_mean': float(loss_ns.detach()),
                'L_ss_final_epoch_mean': float(np.mean(ss_losses)) if ss_losses else None,
                'L_align_final_epoch_mean': float(align_loss.detach()),
                'num_negatives_requested': args.num_negatives,
                'num_negatives_effective': neg_stats.get('num_negatives_effective'),
                'negative_fallback_reason': neg_stats.get('negative_fallback_reason'),
                'intra_neg_ratio_requested': args.intra_neg_ratio,
                'intra_neg_ratio_effective': neg_stats.get('intra_neg_ratio_effective'),
                'ego_exclusion_enabled': neg_stats.get('ego_exclusion_enabled'),
                'avg_excluded_ego_size': neg_stats.get('avg_excluded_ego_size'),
                'fallback_reason': neg_stats.get('fallback_reason'),
                'neg_exclude_scope': args.neg_exclude_scope,
                'neg_refresh_interval': args.neg_refresh_interval,
            }
        model.eval()
        head.eval()
        with torch.no_grad():
            h = head.encode(model.gcn, target_feature, target_adj, sparse=sparse)
            g, _ = head.subgraph_embeddings(h, target_adj, k=args.subgraph_hop)
        return h, g, loss_stats, time.time() - start

    for seed in args.formal_seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        feature_embeds = target_feature
        no_pretrain_model = PrePrompt(
            args.unify_dim,
            args.hid_units,
            'prelu',
            max(1, len(source_graphs)),
            args.layers_num,
            0.1,
            type_=args.combinetype,
            backbone=args.backbone,
            alpha=args.alpha,
            ablation=args.ablation_pre,
        ).to(device)
        no_pretrain_model.eval()
        with torch.no_grad():
            no_pretrain_h = no_pretrain_model.gcn(target_feature, target_adj, sparse, False).squeeze(0)

        sam_h, sam_loss_stats, sam_time = train_samgpt(seed)
        usp_h, usp_g, usp_loss_stats, usp_time = train_usp(seed, 'USPFull', args.usp_lambda_ns, args.usp_lambda_ss)
        nosub_h, nosub_g, nosub_loss_stats, nosub_time = train_usp(seed, 'USPNoSubgraphObjective', 0.0, 0.0)

        for shot in args.shots:
            eval_embeddings('FeatureOnlyPrototype', 'none', feature_embeds, feature_embeds, 'node', shot, seed)
            eval_embeddings('NoPretrainGCNPrototype', 'none', no_pretrain_h, no_pretrain_h, 'node', shot, seed)
            eval_embeddings('SAMGPTPrototypeEval', 'GRAPHCL', sam_h, sam_h, 'node', shot, seed, sam_time, sam_loss_stats)
            eval_embeddings('USPFull', 'USP', usp_h, usp_g, args.query_mode, shot, seed, usp_time, usp_loss_stats)
            eval_embeddings(
                'USPNoSubgraphObjective',
                'USP',
                nosub_h,
                nosub_g,
                args.query_mode,
                shot,
                seed,
                nosub_time,
                nosub_loss_stats,
            )

    _write_rows_csv(table_name, rows)
    _write_formal_batch1_summary(summary_name, rows)
    return table_name


