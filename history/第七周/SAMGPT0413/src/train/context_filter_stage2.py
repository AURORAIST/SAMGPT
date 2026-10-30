import os

import numpy as np
import torch

from train.stage_pipeline import get_stage1_reference_graph, split_graph_to_subgraphs


class Stage2GraphFilter:
    """根据 stage2 的 learnability 分数，对图中的节点进行筛选。"""

    def __init__(
        self,
        *,
        parent_directory,
        stage2_outputs_root,
        max_subgraph_nodes,
        stage1_subgraph_hop_n,
        stage1_remove_selfloops,
        stage2_top_ratio,
        torch_load_trusted,
    ):
        # 项目根目录
        self.parent_directory = parent_directory

        # stage2 输出目录根路径
        self.stage2_outputs_root = stage2_outputs_root

        # 每个子图允许的最大节点数
        self.max_subgraph_nodes = int(max_subgraph_nodes)

        # 构建子图时使用的 hop 数
        self.stage1_subgraph_hop_n = int(stage1_subgraph_hop_n)

        # 是否在 stage1 参考图中移除自环
        self.stage1_remove_selfloops = bool(stage1_remove_selfloops)

        # 从非代表池子图中，按排序额外保留的比例
        self.stage2_top_ratio = float(stage2_top_ratio)

        # 加载本地可信 .pt 文件的方法
        self.torch_load_trusted = torch_load_trusted

    @staticmethod
    def _merge_pool_and_rank_with_quota(pool_idx, rank_idx, rank_base_k):
        """
        将“代表池中的样本索引”和“按分数排序的样本索引”进行合并。

        规则：
        1. 先保留 pool_idx 中的所有唯一索引；
        2. 再按 rank_idx 的顺序往后补充；
        3. 至少尝试消费 rank_base_k 个排序候选；
        4. 最终目标唯一数 = len(pool_unique) + rank_base_k。

        返回：
        - selected_unique: 最终选中的唯一 sample 索引
        - rank_consumed: 从 rank_idx 中实际遍历/消费过的索引
        - target_unique: 目标希望达到的唯一样本数
        """
        # 先对 pool_idx 去重，同时保持原始顺序
        pool_unique = []
        seen_pool = set()
        for x in pool_idx:
            xi = int(x)
            if xi not in seen_pool:
                seen_pool.add(xi)
                pool_unique.append(xi)

        # 初始选中集合 = 代表池中的样本
        selected_unique = list(pool_unique)
        selected_set = set(selected_unique)

        # 记录从 rank_idx 中消耗过哪些索引
        rank_consumed = []

        # 最终目标：代表池样本数 + 额外从排序中补的样本数
        target_unique = len(pool_unique) + max(0, int(rank_base_k))

        # 如果不需要额外补充，直接返回
        if int(rank_base_k) <= 0:
            return selected_unique, rank_consumed, target_unique

        # 依次从排序列表中取样本补入
        for idx in rank_idx:
            idx = int(idx)
            rank_consumed.append(idx)

            # 只有没被选中过的样本才加入最终集合
            if idx not in selected_set:
                selected_set.add(idx)
                selected_unique.append(idx)

            # 满足两个条件后停止：
            # 1) 至少已经消费了 rank_base_k 个排序样本
            # 2) 最终唯一样本数已经达到目标
            if len(rank_consumed) >= max(0, int(rank_base_k)) and len(selected_unique) >= target_unique:
                break

        return selected_unique, rank_consumed, target_unique

    def maybe_filter(self, dataset_name, step_id, feature, adj):
        """
        如果存在 stage2 结果文件，则根据 learnability 分数筛选节点；
        如果不存在或读取失败，则直接返回原始 feature 和 adj。

        参数：
        - dataset_name: 数据集名称
        - step_id: 当前图的 graph_id
        - feature: 当前图的节点特征张量，shape=[N, F]
        - adj: 当前图的邻接矩阵（通常为 scipy sparse matrix）

        返回：
        - feature: 过滤后的特征
        - adj: 过滤后的邻接矩阵
        - filtered_flag: 是否实际执行了过滤
        - node_idx: 被保留节点的原始索引；若未过滤则为 None
        """
        # 当前数据集对应的 stage2 输出目录
        stage2_dir = os.path.join(self.stage2_outputs_root, dataset_name.lower())

        # stage2 学习性分数文件
        score_pt_path = os.path.join(stage2_dir, 'learnability_scores.pt')

        # 代表池文件
        pool_pt_path = os.path.join(stage2_dir, 'representative_pool.pt')

        # 原始图的节点数
        original_nodes = int(feature.size(0))

        # 如果分数文件不存在，则跳过过滤
        if not os.path.exists(score_pt_path):
            print(f'[stage2] skip {dataset_name}: score pt not found at {score_pt_path}')
            return feature, adj, False, None

        # 读取分数文件
        try:
            score_pt = self.torch_load_trusted(score_pt_path)
        except Exception as e:
            print(f'[stage2] skip {dataset_name}: failed to load score pt: {e}')
            return feature, adj, False, None

        # 从分数文件中提取必要字段
        graph_ids = score_pt.get('graph_id', None)                    # 每个样本属于哪个图
        subgraph_ids = score_pt.get('subgraph_id', None)             # 每个样本对应哪个子图
        rank_desc = score_pt.get('rank_desc', None)                  # 样本按 learnability 排序后的索引（降序）
        in_pool = score_pt.get('in_representative_pool', None)       # 是否属于 representative pool

        # 缺字段则跳过
        if graph_ids is None or subgraph_ids is None or rank_desc is None or in_pool is None:
            print(f'[stage2] skip {dataset_name}: score pt missing required keys')
            return feature, adj, False, None

        # 转成 numpy，方便后续索引和筛选
        graph_ids = graph_ids.cpu().numpy().astype(np.int64)
        subgraph_ids = subgraph_ids.cpu().numpy().astype(np.int64)
        rank_desc = rank_desc.cpu().numpy().astype(np.int64)
        in_pool = in_pool.cpu().numpy().astype(np.int64)

        # 找到当前 graph_id=step_id 对应的所有样本位置
        valid_idx = np.where(graph_ids == int(step_id))[0]

        # 当前图在 stage2 中没有样本，则跳过
        if valid_idx.size == 0:
            print(f'[stage2] skip {dataset_name}: no stage2 samples for graph_id={step_id}')
            return feature, adj, False, None

        # 记录代表池中、属于当前图的 sample 索引
        pool_selected_sample_idx = []

        # 如果 representative_pool.pt 存在，则优先使用它来确定 pool 中的样本
        if os.path.exists(pool_pt_path):
            try:
                pool_pt = self.torch_load_trusted(pool_pt_path)
                pool_graph = pool_pt.get('pool_graph_id', None)
                pool_subgraph = pool_pt.get('pool_subgraph_id', None)

                if pool_graph is not None and pool_subgraph is not None:
                    pool_graph = pool_graph.cpu().numpy().astype(np.int64)
                    pool_subgraph = pool_subgraph.cpu().numpy().astype(np.int64)

                    # 遍历 pool 文件中记录的 (graph_id, subgraph_id)
                    # 找到与当前 step_id 对应的样本位置
                    for g_id, sg_id in zip(pool_graph.tolist(), pool_subgraph.tolist()):
                        if g_id == int(step_id):
                            hit = np.where((graph_ids == g_id) & (subgraph_ids == sg_id))[0]
                            for h in hit.tolist():
                                pool_selected_sample_idx.append(int(h))
            except Exception as e:
                print(f'[stage2] warn {dataset_name}: failed to load pool pt: {e}')
        else:
            # 如果没有单独的 representative_pool.pt，
            # 就直接使用 learnability_scores.pt 里的 in_representative_pool 字段
            for i in valid_idx.tolist():
                if in_pool[i] == 1:
                    pool_selected_sample_idx.append(int(i))

        # stage2_top_ratio 做范围裁剪，保证在 [0, 1] 之间
        top_ratio = max(0.0, min(1.0, self.stage2_top_ratio))

        # 代表池中的唯一 sample 数量
        pool_unique_count = len(set(pool_selected_sample_idx))

        # 当前图总样本数 - 代表池样本数 = 剩余可从排序中选择的样本基数
        remaining_count = max(0, int(valid_idx.size) - int(pool_unique_count))

        # 从排序结果中额外保留多少个样本
        keep_count = int(np.ceil(remaining_count * top_ratio))

        # 取出当前图对应的所有排序样本（按 rank_desc 的顺序）
        ranked_valid = [idx for idx in rank_desc.tolist() if graph_ids[idx] == int(step_id)]

        # 合并“代表池样本”和“按排名选出的额外样本”
        selected_sample_idx, rank_selected_sample_idx, target_unique = self._merge_pool_and_rank_with_quota(
            pool_idx=pool_selected_sample_idx,
            rank_idx=ranked_valid,
            rank_base_k=keep_count,
        )

        # 如果合并后没有选中任何样本，则跳过
        if len(selected_sample_idx) == 0:
            print(f'[stage2] skip {dataset_name}: no selected samples after filtering')
            return feature, adj, False, None

        # 读取当前图对应的 stage1 参考图
        g = get_stage1_reference_graph(
            dataset_name=dataset_name,
            data_root=os.path.join(self.parent_directory, 'data'),
            graph_id=int(step_id),
            remove_selfloops=self.stage1_remove_selfloops,
        )

        # 将参考图切分为一组子图，保证子图编号与 stage2 中的 subgraph_id 对齐
        subgraphs = split_graph_to_subgraphs(
            g,
            max_nodes=self.max_subgraph_nodes,
            hop_n=self.stage1_subgraph_hop_n,
        )

        # 最终保留的节点集合
        selected_nodes = set()

        # 分别统计来自代表池和来自排序补充的节点集合
        pool_selected_nodes = set()
        rank_selected_nodes = set()

        # 转集合，加快判断效率
        pool_selected_sample_idx_set = set(pool_selected_sample_idx)
        rank_selected_sample_idx_set = set(rank_selected_sample_idx)

        # 根据选中的 sample -> subgraph_id -> 子图节点，汇总出最终保留节点
        for idx in selected_sample_idx:
            sg_id = int(subgraph_ids[idx])

            # 确保子图编号合法
            if 0 <= sg_id < len(subgraphs):
                sg_nodes = set(subgraphs[sg_id].nodes())

                # 加入总节点集合
                selected_nodes.update(sg_nodes)

                # 分别记录 pool / rank 对应的节点
                if idx in pool_selected_sample_idx_set:
                    pool_selected_nodes.update(sg_nodes)
                if idx in rank_selected_sample_idx_set:
                    rank_selected_nodes.update(sg_nodes)

        # 统计来自 pool 和 rank 的节点重叠数量
        node_overlap_count = len(pool_selected_nodes.intersection(rank_selected_nodes))

        # 如果最终节点数太少，不值得过滤，直接跳过
        if len(selected_nodes) < 2:
            print(f'[stage2] skip {dataset_name}: selected nodes too few ({len(selected_nodes)})')
            return feature, adj, False, None

        # 节点索引排序后转 numpy
        node_idx = np.array(sorted(selected_nodes), dtype=np.int64)

        # 转成 torch 索引张量
        node_idx_t = torch.tensor(node_idx, dtype=torch.long)

        # 按节点索引筛选特征
        feature = feature.index_select(0, node_idx_t)

        # 按节点索引筛选邻接矩阵
        adj = adj[node_idx][:, node_idx]

        # 打印过滤统计信息
        print(
            f'[stage2] {dataset_name} graph_id={step_id}: '
            f'selected_subgraphs={len(selected_sample_idx)}, '
            f'target_unique={target_unique}, '
            f'rank_consumed={len(rank_selected_sample_idx)}, '
            f'pool_nodes={len(pool_selected_nodes)}, '
            f'rank_nodes={len(rank_selected_nodes)}, '
            f'node_overlap={node_overlap_count}, '
            f'selected_nodes={len(node_idx)}, '
            f'original_nodes={original_nodes}, '
            f'keep_ratio={len(node_idx) / max(1, original_nodes):.4f}'
        )

        # 返回过滤后的图
        return feature, adj, True, node_idx
