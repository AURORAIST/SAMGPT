import os
import shutil
import csv
import json

import numpy as np
import torch
from sklearn.decomposition import PCA

# 构建代表性子图池（stage2）
from layers.stage2_pool_builder import build_representative_pool
# stage1：GMM 聚类
from models.stage1_gmm_cluster import fit_clusters
# stage2：可学习性计算
from models.stage2_learnability_model import FEATURE_KEYS, compute_learnability
# 子图编码与切分
from train.stage_pipeline import encode_subgraph, get_stage1_reference_graph, split_graph_to_subgraphs
# 数据加载（当前文件未直接用）
from utils.dataset import load_dataset


def _row_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return x / norm


def _softmax_np(logits, axis=1):
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    denom = np.sum(exp_logits, axis=axis, keepdims=True)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return exp_logits / denom


def pca_compression_subgraph(feature_np, out_dim):
    """对子图特征做 PCA 降维到 out_dim，当子图过小时用 0 填充"""
    in_dim = int(feature_np.shape[1])   # 输入特征维度
    n_nodes = int(feature_np.shape[0])  # 子图节点数

    # 有效维度不能超过：输出维度、输入维度、节点数
    eff_dim = int(min(out_dim, in_dim, n_nodes))

    if eff_dim <= 0:
        return torch.zeros((n_nodes, out_dim), dtype=torch.float32)

    # PCA降维
    pca = PCA(n_components=eff_dim)
    compressed = pca.fit_transform(feature_np)
    compressed = torch.FloatTensor(compressed)

    # 如果维度不足，补0
    if eff_dim == out_dim:
        return compressed

    pad = torch.zeros((n_nodes, out_dim - eff_dim), dtype=torch.float32)
    return torch.cat([compressed, pad], dim=1)


class GlobalSubgraphPoolBuilder:
    """构建跨数据集的全局子图池，并执行stage2筛选"""

    def __init__(
        self,
        *,
        parent_directory,
        cache_dir,
        stage1_outputs_root,
        stage2_outputs_root,
        max_subgraph_nodes,
        stage1_subgraph_hop_n,
        stage1_num_clusters,
        stage1_encoding_mode,
        stage1_feature_k,
        stage1_feature_mask,
        stage1_use_log1p,
        stage1_remove_selfloops,
        stage2_top_m_per_cluster,
        stage2_alpha,
        stage2_beta,
        stage2_gamma,
        stage2_top_ratio,
        seed,
        force_rebuild_global_pool,
        torch_load_trusted,
        evaluate_stage1_artifacts,
        evaluate_stage2_artifacts,
    ):
        # 路径配置
        self.parent_directory = parent_directory
        self.cache_dir = cache_dir
        self.stage1_outputs_root = stage1_outputs_root
        self.stage2_outputs_root = stage2_outputs_root

        # 子图构建参数
        self.max_subgraph_nodes = int(max_subgraph_nodes)
        self.stage1_subgraph_hop_n = int(stage1_subgraph_hop_n)

        # stage1（聚类）参数
        self.stage1_num_clusters = int(stage1_num_clusters)
        self.stage1_encoding_mode = str(stage1_encoding_mode)
        self.stage1_feature_k = int(stage1_feature_k)
        self.stage1_feature_mask = str(stage1_feature_mask)
        self.stage1_use_log1p = bool(stage1_use_log1p)
        self.stage1_remove_selfloops = bool(stage1_remove_selfloops)

        # stage2（筛选）参数
        self.stage2_top_m_per_cluster = int(stage2_top_m_per_cluster)
        self.stage2_alpha = float(stage2_alpha)
        self.stage2_beta = float(stage2_beta)
        self.stage2_gamma = float(stage2_gamma)
        self.stage2_top_ratio = float(stage2_top_ratio)

        # 其他
        self.seed = int(seed)
        self.force_rebuild_global_pool = int(force_rebuild_global_pool)
        self.torch_load_trusted = torch_load_trusted

        # 评估函数
        self.evaluate_stage1_artifacts = evaluate_stage1_artifacts
        self.evaluate_stage2_artifacts = evaluate_stage2_artifacts

    def build(self, datasets, target_graph_ids_zero_based):
        """主流程：构建全局子图池"""

        # ========= 路径 =========
        cache_dir_global = os.path.join(self.cache_dir, 'global_pool')
        os.makedirs(cache_dir_global, exist_ok=True)

        cache_path = os.path.join(cache_dir_global, 'global_subgraph_pool.pt')

        run_stage1_dir = os.path.join(self.stage1_outputs_root, 'global_merged')
        run_stage2_dir = os.path.join(self.stage2_outputs_root, 'global_merged')

        # 固定缓存 artifacts（用于恢复）
        global_artifacts_root = os.path.join(cache_dir_global, 'artifacts')
        global_stage1_dir_fixed = os.path.join(global_artifacts_root, 'stage1', 'global_merged')
        global_stage2_dir_fixed = os.path.join(global_artifacts_root, 'stage2', 'global_merged')

        global_stage1_csv_fixed = os.path.join(global_stage1_dir_fixed, 'structural_encodings.csv')
        global_stage1_summary_fixed = os.path.join(global_stage1_dir_fixed, 'stage1_summary.json')
        global_stage2_csv_fixed = os.path.join(global_stage2_dir_fixed, 'learnability_scores.csv')

        os.makedirs(run_stage1_dir, exist_ok=True)
        os.makedirs(run_stage2_dir, exist_ok=True)

        def _write_global_stage_csvs():
            stage1_csv = os.path.join(run_stage1_dir, 'structural_encodings.csv')
            with open(stage1_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'graph_id', 'subgraph_id'] + list(FEATURE_KEYS) + ['cluster_id'])
                for i, rec in enumerate(merged_records):
                    writer.writerow([
                        int(rec['sample_id']),
                        int(rec['graph_id']),
                        int(rec['subgraph_id']),
                        float(rec['num_nodes']),
                        float(rec['num_edges']),
                        float(rec['avg_degree']),
                        float(rec['max_degree']),
                        float(rec['var_degree']),
                        float(rec['density']),
                        float(rec['avg_clustering']),
                        float(rec['triangles']),
                        float(rec['diameter']),
                        int(cluster_ids[i]),
                    ])

            stage1_summary = {
                'encoding_mode': self.stage1_encoding_mode,
                'feature_k': int(self.stage1_feature_k),
                'feature_mask': self.stage1_feature_mask,
                'cluster_feature_names': list(cluster_feature_names),
                'num_clusters': int(self.stage1_num_clusters),
                'num_samples': int(len(merged_records)),
                'datasets': list(datasets),
                'target_graph_ids': [int(x) for x in target_graph_ids_zero_based],
            }
            stage1_summary_path = os.path.join(run_stage1_dir, 'stage1_summary.json')
            with open(stage1_summary_path, 'w', encoding='utf-8') as f:
                json.dump(stage1_summary, f, ensure_ascii=False, indent=2)

            stage2_csv = os.path.join(run_stage2_dir, 'learnability_scores.csv')
            with open(stage2_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'sample_id',
                    'graph_id',
                    'subgraph_id',
                    'cluster_id',
                    'novelty',
                    'coverage',
                    'redundancy',
                    'novelty_norm',
                    'coverage_norm',
                    'redundancy_norm',
                    'learnability',
                    'rank',
                    'in_representative_pool',
                ])

                rank_arr = np.empty(len(order), dtype=np.int64)
                rank_arr[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
                in_pool_set = set(int(i) for i in rep_indices.tolist())

                for i, rec in enumerate(merged_records):
                    writer.writerow([
                        int(rec['sample_id']),
                        int(rec['graph_id']),
                        int(rec['subgraph_id']),
                        int(cluster_ids[i]),
                        float(novelty[i]),
                        float(coverage[i]),
                        float(redundancy[i]),
                        float(novelty_n[i]),
                        float(coverage_n[i]),
                        float(redundancy_n[i]),
                        float(learnability[i]),
                        int(rank_arr[i]),
                        int(i in in_pool_set),
                    ])

            # 固定 artifacts 路径也落一份，供缓存命中时恢复
            os.makedirs(global_stage1_dir_fixed, exist_ok=True)
            os.makedirs(global_stage2_dir_fixed, exist_ok=True)
            shutil.copy2(stage1_csv, global_stage1_csv_fixed)
            shutil.copy2(stage1_summary_path, global_stage1_summary_fixed)
            shutil.copy2(stage2_csv, global_stage2_csv_fixed)

        # ========= 缓存签名（用于判断是否可以复用） =========
        cache_signature = {
            'datasets': list(datasets),
            'target_graph_ids_zero_based': [int(x) for x in target_graph_ids_zero_based],
            'max_subgraph_nodes': self.max_subgraph_nodes,
            'stage1_subgraph_hop_n': self.stage1_subgraph_hop_n,
            'stage1_num_clusters': self.stage1_num_clusters,
            'stage1_encoding_mode': self.stage1_encoding_mode,
            'stage1_feature_k': self.stage1_feature_k,
            'stage1_feature_mask': self.stage1_feature_mask,
            'stage1_use_log1p': int(self.stage1_use_log1p),
            'stage1_remove_selfloops': int(self.stage1_remove_selfloops),
            'stage2_top_m_per_cluster': self.stage2_top_m_per_cluster,
            'stage2_alpha': self.stage2_alpha,
            'stage2_beta': self.stage2_beta,
            'stage2_gamma': self.stage2_gamma,
            'stage2_top_ratio': self.stage2_top_ratio,
            'stage2_allow_duplicates': 1,
            'seed': self.seed,
        }

        # ========= 尝试加载缓存 =========
        if self.force_rebuild_global_pool != 1:
            if os.path.exists(cache_path):
                try:
                    cached = self.torch_load_trusted(cache_path)

                    # 校验签名一致
                    if isinstance(cached, dict) and cached.get('signature', None) == cache_signature:
                        selected_samples = cached.get('selected_samples', [])
                        cluster_similarity = np.asarray(
                            cached.get('cluster_similarity', None), dtype=np.float64
                        )

                        # 成功加载缓存
                        if len(selected_samples) > 0 and cluster_similarity is not None:
                            print(f'[global_pool] loaded cache: {cache_path} '
                                  f'(selected={len(selected_samples)})')

                            # 恢复CSV文件（用于评估）
                            try:
                                if os.path.exists(global_stage1_csv_fixed):
                                    os.makedirs(run_stage1_dir, exist_ok=True)
                                    shutil.copy2(
                                        global_stage1_csv_fixed,
                                        os.path.join(run_stage1_dir, 'structural_encodings.csv'),
                                    )
                                if os.path.exists(global_stage1_summary_fixed):
                                    os.makedirs(run_stage1_dir, exist_ok=True)
                                    shutil.copy2(
                                        global_stage1_summary_fixed,
                                        os.path.join(run_stage1_dir, 'stage1_summary.json'),
                                    )

                                if os.path.exists(global_stage2_csv_fixed):
                                    os.makedirs(run_stage2_dir, exist_ok=True)
                                    shutil.copy2(
                                        global_stage2_csv_fixed,
                                        os.path.join(run_stage2_dir, 'learnability_scores.csv'),
                                    )

                                stage1_eval = self.evaluate_stage1_artifacts(run_stage1_dir)
                                print(f"[stage1-metrics] global_merged: {stage1_eval['metrics_json']}")

                            except Exception as e:
                                print(f"[stage1-metrics] warn global_merged(cache): {e}")

                            try:
                                stage2_eval = self.evaluate_stage2_artifacts(run_stage2_dir)
                                print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")

                            except Exception as e:
                                print(f"[stage2-metrics] warn global_merged(cache): {e}")

                            return selected_samples, cluster_similarity

                except Exception as e:
                    print(f'[global_pool] warn: failed to load cache {cache_path}: {e}')

        # ========= 构建子图 =========
        merged_records = []   # 存特征
        merged_meta = []      # 存元信息
        sample_id = 0

        dataset_subgraph_counts = {ds: 0 for ds in datasets}

        for dataset_idx, ds in enumerate(datasets):
            for graph_id in target_graph_ids_zero_based:

                # 获取原始图
                try:
                    g = get_stage1_reference_graph(
                        dataset_name=ds,
                        data_root=os.path.join(self.parent_directory, 'data'),
                        graph_id=int(graph_id),
                        remove_selfloops=self.stage1_remove_selfloops,
                    )
                except Exception as e:
                    print(f'[global_pool] skip {ds} graph_id={graph_id}: {e}')
                    continue

                # 切分子图
                subgraphs = split_graph_to_subgraphs(
                    g,
                    max_nodes=self.max_subgraph_nodes,
                    hop_n=self.stage1_subgraph_hop_n,
                )

                # 编码子图
                for subgraph_id, subg in enumerate(subgraphs):
                    encoded = encode_subgraph(subg)

                    # 结构特征
                    rec = {
                        'sample_id': sample_id,
                        'graph_id': int(graph_id),
                        'subgraph_id': int(subgraph_id),
                    }
                    rec.update(encoded)

                    merged_records.append(rec)

                    # 元信息（用于回溯）
                    merged_meta.append(
                        {
                            'sample_id': sample_id,
                            'dataset_name': ds,
                            'dataset_idx': int(dataset_idx),
                            'graph_id': int(graph_id),
                            'subgraph_id': int(subgraph_id),
                            'node_idx': np.array(sorted(list(subg.nodes())), dtype=np.int64),
                        }
                    )

                    sample_id += 1
                    dataset_subgraph_counts[ds] += 1

        if len(merged_records) == 0:
            raise RuntimeError('Global subgraph pool is empty.')

        print(f"[debug] subgraph build done: total_subgraphs={len(merged_records)}, "
              f"per_dataset={dataset_subgraph_counts}")

        # ========= Stage1：聚类 =========
        labels, centers_raw, _, _, _, cluster_feature_names = fit_clusters(
            records=merged_records,
            num_clusters=self.stage1_num_clusters,
            seed=self.seed,
            use_log1p=self.stage1_use_log1p,
            encoding_mode=self.stage1_encoding_mode,
            feature_k=self.stage1_feature_k,
            feature_mask=self.stage1_feature_mask,
        )

        # ========= 构造特征矩阵 =========
        x = np.column_stack([
            np.array([float(r[k]) for r in merged_records], dtype=np.float64)
            for k in FEATURE_KEYS
        ])

        cluster_ids = labels.astype(np.int64)

        # ========= Stage2：代表性子图 =========
        rep_indices, _ = build_representative_pool(
            x=x,
            cluster_ids=cluster_ids,
            top_m_per_cluster=self.stage2_top_m_per_cluster,
            centers_raw=centers_raw,
        )

        # ========= Stage2：可学习性 =========
        novelty, coverage, redundancy, \
        novelty_n, coverage_n, redundancy_n, \
        learnability, order = compute_learnability(
            x=x,
            cluster_ids=cluster_ids,
            rep_indices=rep_indices,
            alpha=self.stage2_alpha,
            beta=self.stage2_beta,
            gamma=self.stage2_gamma,
        )

        # ========= 构建最终样本池 =========
        pool_selected_idx = [int(i) for i in rep_indices.tolist()]

        # 去重
        pool_unique = []
        seen_pool = set()
        for pid in pool_selected_idx:
            if pid not in seen_pool:
                seen_pool.add(pid)
                pool_unique.append(pid)

        selected_idx = list(pool_unique)
        selected_set = set(selected_idx)

        # 按 learnability 补充样本
        top_ratio = max(0.0, min(1.0, self.stage2_top_ratio))
        remaining_count = len(merged_records) - len(pool_unique)
        keep_count = int(np.ceil(remaining_count * top_ratio))

        target_unique = len(pool_unique) + keep_count

        rank_selected_idx = []

        if keep_count > 0:
            for idx in order.tolist():
                idx = int(idx)
                rank_selected_idx.append(idx)

                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_idx.append(idx)

                if len(rank_selected_idx) >= keep_count and len(selected_idx) >= target_unique:
                    break

        print(f"[debug] sample pool done: representative_pool={len(rep_indices)}, "
              f"selected={len(selected_idx)}")

        # ========= cluster 相似度 =========
        center_norm = np.linalg.norm(centers_raw, axis=1, keepdims=True)
        center_unit = centers_raw / np.where(center_norm < 1e-12, 1.0, center_norm)
        cluster_similarity = center_unit @ center_unit.T

        # ========= 共享结构原型字典工件 =========
        # 使用 stage1 聚类中心作为原型中心；样本到原型软分配来自余弦相似度 softmax。
        x_unit = _row_normalize(x)
        logits = x_unit @ center_unit.T
        sample_to_prototype = _softmax_np(logits, axis=1).astype(np.float32)

        # ========= 构建输出 =========
        selected_samples = []
        for idx in selected_idx:
            meta = merged_meta[idx]
            proto_weights = sample_to_prototype[int(idx)].astype(np.float32)
            proto_id = int(np.argmax(proto_weights))
            selected_samples.append(
                {
                    'sample_id': meta['sample_id'],
                    'dataset_name': meta['dataset_name'],
                    'dataset_idx': int(meta['dataset_idx']),
                    'graph_id': meta['graph_id'],
                    'subgraph_id': meta['subgraph_id'],
                    'node_idx': meta['node_idx'],
                    'cluster_id': int(cluster_ids[idx]),
                    'prototype_id': proto_id,
                    'prototype_weights': proto_weights,
                    'learnability': float(learnability[idx]),
                }
            )

        # ========= 保存缓存 =========
        cache_payload = {
            'signature': cache_signature,
            'selected_samples': selected_samples,
            'cluster_similarity': cluster_similarity,
            'prototype_centers': centers_raw.astype(np.float32),
            'sample_to_prototype': sample_to_prototype,
        }

        try:
            _write_global_stage_csvs()
            stage1_eval = self.evaluate_stage1_artifacts(run_stage1_dir)
            print(f"[stage1-metrics] global_merged: {stage1_eval['metrics_json']}")
        except Exception as e:
            print(f"[stage1-metrics] warn global_merged(build): {e}")

        try:
            stage2_eval = self.evaluate_stage2_artifacts(run_stage2_dir)
            print(f"[stage2-metrics] global_merged: {stage2_eval['metrics_json']}")
        except Exception as e:
            print(f"[stage2-metrics] warn global_merged(build): {e}")

        try:
            torch.save(cache_payload, cache_path)
            print(f'[global_pool] saved cache: {cache_path}')
        except Exception as e:
            print(f'[global_pool] warn: failed to save cache {cache_path}: {e}')

        # 额外导出可审计的原型字典工件
        os.makedirs(run_stage1_dir, exist_ok=True)
        os.makedirs(global_stage1_dir_fixed, exist_ok=True)

        run_proto_centers = os.path.join(run_stage1_dir, 'prototype_centers.npy')
        run_proto_sim = os.path.join(run_stage1_dir, 'prototype_similarity.npy')
        run_sample2proto = os.path.join(run_stage1_dir, 'sample_to_prototype.npy')

        fixed_proto_centers = os.path.join(global_stage1_dir_fixed, 'prototype_centers.npy')
        fixed_proto_sim = os.path.join(global_stage1_dir_fixed, 'prototype_similarity.npy')
        fixed_sample2proto = os.path.join(global_stage1_dir_fixed, 'sample_to_prototype.npy')

        np.save(run_proto_centers, centers_raw.astype(np.float32))
        np.save(run_proto_sim, cluster_similarity.astype(np.float32))
        np.save(run_sample2proto, sample_to_prototype)

        np.save(fixed_proto_centers, centers_raw.astype(np.float32))
        np.save(fixed_proto_sim, cluster_similarity.astype(np.float32))
        np.save(fixed_sample2proto, sample_to_prototype)

        required_paths = [
            run_proto_centers,
            run_proto_sim,
            run_sample2proto,
            fixed_proto_centers,
            fixed_proto_sim,
            fixed_sample2proto,
        ]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f'prototype artifact write failed, missing files: {missing}')

        return selected_samples, cluster_similarity
