import os

from train.stage_pipeline import Stage1Config, Stage2Config, run_stage1, run_stage2


class ContextArtifactsManager:
    """用于构建/刷新数据集的 stage1 和 stage2 中间产物"""

    def __init__(
        self,
        *,
        parent_directory,            # 项目根目录
        stage1_outputs_root,         # stage1 输出目录根路径
        stage2_outputs_root,         # stage2 输出目录根路径
        max_subgraph_nodes,          # 子图最大节点数
        stage1_subgraph_hop_n,       # 子图 hop 数
        stage1_num_clusters,         # stage1 聚类数量
        stage1_encoding_mode,        # 编码模式
        stage1_feature_k,            # 特征维度 k
        stage1_feature_mask,         # raw9 维度掩码
        seed,                        # 随机种子
        stage1_remove_selfloops,     # 是否移除自环
        stage1_use_log1p,            # 是否使用 log1p 变换
        force_rebuild_stage1,        # 是否强制重建 stage1
        force_rebuild_stage2,        # 是否强制重建 stage2
        stage2_top_m_per_cluster,    # 每个 cluster 选取 top-m
        stage2_alpha,                # stage2 参数 alpha
        stage2_beta,                 # stage2 参数 beta
        stage2_gamma,                # stage2 参数 gamma
        evaluate_stage1_artifacts,   # stage1 评估函数
        evaluate_stage2_artifacts,   # stage2 评估函数
    ):
        self.parent_directory = parent_directory
        self.stage1_outputs_root = stage1_outputs_root
        self.stage2_outputs_root = stage2_outputs_root
        self.max_subgraph_nodes = int(max_subgraph_nodes)
        self.stage1_subgraph_hop_n = int(stage1_subgraph_hop_n)
        self.stage1_num_clusters = int(stage1_num_clusters)
        self.stage1_encoding_mode = str(stage1_encoding_mode)
        self.stage1_feature_k = int(stage1_feature_k)
        self.stage1_feature_mask = str(stage1_feature_mask)
        self.seed = int(seed)
        self.stage1_remove_selfloops = bool(stage1_remove_selfloops)
        self.stage1_use_log1p = bool(stage1_use_log1p)
        self.force_rebuild_stage1 = int(force_rebuild_stage1)
        self.force_rebuild_stage2 = int(force_rebuild_stage2)
        self.stage2_top_m_per_cluster = int(stage2_top_m_per_cluster)
        self.stage2_alpha = float(stage2_alpha)
        self.stage2_beta = float(stage2_beta)
        self.stage2_gamma = float(stage2_gamma)
        self.evaluate_stage1_artifacts = evaluate_stage1_artifacts
        self.evaluate_stage2_artifacts = evaluate_stage2_artifacts

    def ensure(self, datasets):
        """确保所有数据集的 stage1 / stage2 产物存在（如果不存在则构建）"""
        stage1_root = self.stage1_outputs_root
        stage2_root = self.stage2_outputs_root

        for ds in datasets:
            ds_lower = ds.lower()  # 数据集名称统一小写

            # =======================
            # Stage 1 部分
            # =======================
            ds_stage1_dir = os.path.join(stage1_root, ds_lower)
            stage1_csv = os.path.join(ds_stage1_dir, 'structural_encodings.csv')

            # 判断是否需要构建 stage1：
            # 1. 强制重建标志为 1
            # 2. 输出文件不存在
            need_build_stage1 = self.force_rebuild_stage1 == 1 or (not os.path.exists(stage1_csv))

            if need_build_stage1:
                print(f'[context] Building stage1 for {ds} -> {ds_stage1_dir}')
                run_stage1(
                    Stage1Config(
                        dataset=ds,
                        data_root=os.path.join(self.parent_directory, 'data'),  # 数据目录
                        output_root=stage1_root,
                        max_subgraph_nodes=self.max_subgraph_nodes,
                        subgraph_hop_n=self.stage1_subgraph_hop_n,
                        num_clusters=self.stage1_num_clusters,
                        encoding_mode=self.stage1_encoding_mode,
                        feature_k=self.stage1_feature_k,
                        feature_mask=self.stage1_feature_mask,
                        seed=self.seed,
                        remove_selfloops=self.stage1_remove_selfloops,
                        use_log1p=self.stage1_use_log1p,
                    )
                )

            # =======================
            # Stage 2 部分
            # =======================
            ds_stage2_dir = os.path.join(stage2_root, ds_lower)
            stage2_pt = os.path.join(ds_stage2_dir, 'learnability_scores.pt')

            # 判断是否需要构建 stage2
            need_build_stage2 = self.force_rebuild_stage2 == 1 or (not os.path.exists(stage2_pt))

            if need_build_stage2:
                # 如果 stage1 结果不存在，无法继续
                if not os.path.exists(stage1_csv):
                    raise FileNotFoundError(
                        f'Cannot build stage2 for {ds}: missing stage1 csv at {stage1_csv}'
                    )

                print(f'[context] Building stage2 for {ds} -> {ds_stage2_dir}')
                run_stage2(
                    Stage2Config(
                        input_csv=stage1_csv,  # 输入为 stage1 输出
                        output_dir=ds_stage2_dir,
                        top_m_per_cluster=self.stage2_top_m_per_cluster,
                        alpha=self.stage2_alpha,
                        beta=self.stage2_beta,
                        gamma=self.stage2_gamma,
                    )
                )

            # =======================
            # Stage 1 评估
            # =======================
            try:
                stage1_eval = self.evaluate_stage1_artifacts(ds_stage1_dir)
                print(f"[stage1-metrics] {ds}: {stage1_eval['metrics_json']}")
            except Exception as e:
                print(f"[stage1-metrics] warn {ds}: {e}")

            # =======================
            # Stage 2 评估
            # =======================
            try:
                stage2_eval = self.evaluate_stage2_artifacts(ds_stage2_dir)
                print(f"[stage2-metrics] {ds}: {stage2_eval['metrics_json']}")
            except Exception as e:
                print(f"[stage2-metrics] warn {ds}: {e}")
