import os

import numpy as np

from train.stage_pipeline import get_stage1_reference_graph, split_graph_to_subgraphs


class StructContextMapper:
    """
    用于加载并缓存“结构上下文”相关信息：

    1. 结构簇之间的相似度矩阵（cluster similarity）
    2. 每个节点所属的结构簇标签（node -> cluster_id）

    这些信息通常由 stage1 / stage2 的中间产物生成，
    后续会被 EGRAPHCL 或结构增强模块使用。
    """

    def __init__(
        self,
        *,
        parent_directory,
        stage1_outputs_root,
        stage2_outputs_root,
        max_subgraph_nodes,
        stage1_subgraph_hop_n,
        stage1_remove_selfloops,
    ):
        # 项目根目录，通常用于定位 data/ 等基础路径
        self.parent_directory = parent_directory

        # stage1 输出目录
        # 里面一般包含 structural_encodings.csv、cluster_centers.npy 等文件
        self.stage1_outputs_root = stage1_outputs_root

        # stage2 输出目录
        # 这里会缓存 egraphcl_cluster_similarity.npy 等结果
        self.stage2_outputs_root = stage2_outputs_root

        # 子图切分时允许的最大节点数
        self.max_subgraph_nodes = int(max_subgraph_nodes)

        # 子图构造时使用的 n-hop 邻域深度
        self.stage1_subgraph_hop_n = int(stage1_subgraph_hop_n)

        # 是否在 stage1 参考图构造时移除自环
        self.stage1_remove_selfloops = bool(stage1_remove_selfloops)

        # 缓存：节点结构标签
        # key: (dataset_name.lower(), step_id)
        # val: np.ndarray 或 None
        self.struct_node_label_cache = {}

        # 缓存：结构簇相似度矩阵
        # key: dataset_name.lower()
        # val: np.ndarray 或 None
        self.struct_cluster_sim_cache = {}

    def get_struct_cluster_similarity(self, dataset_name):
        """
        获取某个数据集的“结构簇相似度矩阵”。

        相似度矩阵的构建方式：
        - 从 stage1 的 cluster_centers.npy 中读取各簇中心
        - 对簇中心做 L2 归一化
        - 计算 cosine similarity（归一化后点积）

        返回：
            sim: shape [num_clusters, num_clusters] 的 numpy 数组
            若失败则返回 None
        """
        key = dataset_name.lower()

        # 如果已经缓存，直接返回，避免重复读盘/计算
        if key in self.struct_cluster_sim_cache:
            return self.struct_cluster_sim_cache[key]

        # 当前数据集在 stage1 / stage2 下的输出目录
        ds_dir = os.path.join(self.stage1_outputs_root, key)
        stage2_dir = os.path.join(self.stage2_outputs_root, key)
        os.makedirs(stage2_dir, exist_ok=True)

        # stage2 中缓存的相似度矩阵文件
        sim_npy = os.path.join(stage2_dir, 'egraphcl_cluster_similarity.npy')

        # 若已有缓存文件，优先直接加载
        if os.path.exists(sim_npy):
            try:
                sim = np.load(sim_npy)
                self.struct_cluster_sim_cache[key] = sim
                return sim
            except Exception as e:
                print(f'[egraphcl] warn {dataset_name}: failed to load cached cluster similarity: {e}')

        # 若没有 stage2 缓存，则从 stage1 的 cluster_centers.npy 重新构建
        center_npy = os.path.join(ds_dir, 'cluster_centers.npy')
        if not os.path.exists(center_npy):
            print(f'[egraphcl] skip {dataset_name}: cluster_centers.npy not found at {center_npy}')
            self.struct_cluster_sim_cache[key] = None
            return None

        try:
            # 读取聚类中心，强制转 float64 提高数值稳定性
            centers = np.asarray(np.load(center_npy), dtype=np.float64)

            # 基本合法性检查：必须是二维矩阵且至少有一个 cluster
            if centers.ndim != 2 or centers.shape[0] == 0:
                self.struct_cluster_sim_cache[key] = None
                return None

            # 计算每个 cluster center 的 L2 范数
            norm = np.linalg.norm(centers, axis=1, keepdims=True)

            # 防止除以 0：若范数极小则替换成 1.0
            norm = np.where(norm < 1e-12, 1.0, norm)

            # 单位化，后续点积即 cosine similarity
            c_norm = centers / norm

            # 计算 cluster 间相似度矩阵
            sim = c_norm @ c_norm.T

            # 保存到 stage2 目录，方便下次直接复用
            np.save(sim_npy, sim)

            # 写入内存缓存
            self.struct_cluster_sim_cache[key] = sim
            return sim

        except Exception as e:
            print(f'[egraphcl] skip {dataset_name}: failed to build cluster similarity: {e}')
            self.struct_cluster_sim_cache[key] = None
            return None

    def get_struct_node_labels(self, dataset_name, num_nodes, step_id):
        """
        获取某个数据集中指定 graph(step_id) 的节点结构标签。

        核心思路：
        1. 从 stage1 的 structural_encodings.csv 中读取：
           - graph_id
           - subgraph_id
           - cluster_id
        2. 找出当前 graph(step_id) 对应的所有子图及其 cluster_id
        3. 重新构造该 graph 的参考图，并按相同规则切分子图
        4. 把每个子图对应的 cluster_id 赋给该子图中的所有节点

        参数：
            dataset_name: 数据集名称
            num_nodes: 当前图的节点总数
            step_id: 当前图 id（通常对应 graph_id）

        返回：
            node_labels: shape [num_nodes] 的 np.ndarray
                         每个位置是对应节点的 cluster_id
                         若某节点未被匹配到，默认值为 -1
                         若失败则返回 None
        """
        cache_key = (dataset_name.lower(), int(step_id))

        # 如果缓存中已有该图的节点标签，则直接返回
        if cache_key in self.struct_node_label_cache:
            return self.struct_node_label_cache[cache_key]

        # stage1 生成的结构编码 CSV
        csv_path = os.path.join(self.stage1_outputs_root, dataset_name.lower(), 'structural_encodings.csv')

        if not os.path.exists(csv_path):
            print(f'[struct] skip {dataset_name}: structural_encodings.csv not found at {csv_path}')
            self.struct_node_label_cache[cache_key] = None
            return None

        try:
            # names=True 表示首行作为字段名读取
            # dtype=None 自动推断字段类型
            arr = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        except Exception as e:
            print(f'[struct] skip {dataset_name}: failed to read csv: {e}')
            self.struct_node_label_cache[cache_key] = None
            return None

        # 若 CSV 没有有效数据
        if arr.size == 0:
            self.struct_node_label_cache[cache_key] = None
            return None

        # 保证即使只有一行，也能按一维结构化数组处理
        arr = np.atleast_1d(arr)

        # 提取需要的列
        graph_ids = arr['graph_id'].astype(np.int64)
        subgraph_ids = arr['subgraph_id'].astype(np.int64)
        cluster_ids = arr['cluster_id'].astype(np.int64)

        # 仅保留当前 graph(step_id) 的记录
        use_rows = np.where(graph_ids == int(step_id))[0]
        if use_rows.size == 0:
            self.struct_node_label_cache[cache_key] = None
            return None

        # 构造映射：subgraph_id -> cluster_id
        subgraph_cluster_map = {}
        for ridx in use_rows.tolist():
            subgraph_cluster_map[int(subgraph_ids[ridx])] = int(cluster_ids[ridx])

        # ---------------------------------------------------------
        # 重新加载该 graph 对应的参考图
        # 注意：这里必须与 stage1 使用同样的图构造规则，
        # 否则 subgraph_id 的对应关系可能不一致
        # ---------------------------------------------------------
        g = get_stage1_reference_graph(
            dataset_name=dataset_name,
            data_root=os.path.join(self.parent_directory, 'data'),
            graph_id=int(step_id),
            remove_selfloops=self.stage1_remove_selfloops,
        )

        # 按 stage1 一致的规则切分成子图
        subgraphs = split_graph_to_subgraphs(
            g,
            max_nodes=self.max_subgraph_nodes,
            hop_n=self.stage1_subgraph_hop_n,
        )

        # 初始化所有节点标签为 -1，表示“未分配”
        node_labels = np.full(int(num_nodes), -1, dtype=np.int64)

        # 遍历每个子图，把对应 cluster_id 赋给子图内所有节点
        for sg_id, sg in enumerate(subgraphs):
            # 如果该子图没有出现在 CSV 映射中，则跳过
            if sg_id not in subgraph_cluster_map:
                continue

            c_id = subgraph_cluster_map[sg_id]

            # sg.nodes() 是当前子图包含的原图节点编号
            # 把这些节点的标签统一设为该子图所属 cluster_id
            node_labels[np.array(list(sg.nodes()), dtype=np.int64)] = c_id

        # 写入缓存
        self.struct_node_label_cache[cache_key] = node_labels
        return node_labels
