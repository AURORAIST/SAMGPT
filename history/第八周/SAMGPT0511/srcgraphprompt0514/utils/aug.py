import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np


def aug_random_mask(input_feature, drop_percent=0.2):
    """随机特征遮蔽（feature masking）。

    将部分节点的特征向量置零，模拟特征缺失/扰动。

    Args:
        input_feature: 节点特征，当前实现假设形状类似 [1, N, F]
                      （下标访问使用了 aug_feature[0][j]）。
        drop_percent: 遮蔽比例（节点维上按比例采样需要 mask 的节点数）。

    Returns:
        aug_feature: 遮蔽后的特征张量，形状与 input_feature 相同。
    """
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    # 随机选择要 mask 的节点 id
    mask_idx = random.sample(node_idx, mask_num)

    # deepcopy：避免原地修改输入
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):
    """随机边扰动：随机删边 + 随机加边。

    约定：输入邻接矩阵是无向图（对称），因此 nonzero 会得到双向边 (i,j) 与 (j,i)。

    实现策略：
    - percent = drop_percent / 2
    - 从现有边中随机删除 add_drop_num 条（按“无向边”计数）
    - 再随机添加 add_drop_num 条新边

    Args:
        input_adj: scipy.sparse 的邻接矩阵（通常 CSR），形状 [N, N]
        drop_percent: 扰动比例

    Returns:
        aug_adj: 扰动后的邻接矩阵（scipy.sparse.csr_matrix）
    """
    percent = drop_percent / 2
    coo = input_adj.tocoo()
    undirected_edges = [(int(r), int(c)) for r, c in zip(coo.row, coo.col) if r < c]
    edge_num = len(undirected_edges)
    add_drop_num = int(edge_num * percent / 2)
    if add_drop_num <= 0:
        return input_adj.copy().tocsr()

    aug_adj = input_adj.tolil(copy=True)
    drop_edges = random.sample(undirected_edges, min(add_drop_num, edge_num))
    for u, v in drop_edges:
        aug_adj[u, v] = 0
        aug_adj[v, u] = 0

    node_num = input_adj.shape[0]
    existing = set(undirected_edges)
    add_edges = set()
    max_trials = max(add_drop_num * 20, 100)
    trials = 0
    while len(add_edges) < add_drop_num and trials < max_trials:
        u = random.randrange(node_num)
        v = random.randrange(node_num)
        trials += 1
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing or (a, b) in add_edges:
            continue
        add_edges.add((a, b))

    for u, v in add_edges:
        aug_adj[u, v] = 1
        aug_adj[v, u] = 1

    return aug_adj.tocsr()


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    """随机丢弃节点（node dropping）。

    从图中采样一部分节点直接移除（特征与邻接矩阵对应行列同时删除）。

    Args:
        input_fea: 节点特征，当前实现假设输入形状 [1, N, F]
        input_adj: scipy.sparse 邻接矩阵 [N, N]
        drop_percent: 丢弃节点比例

    Returns:
        aug_input_fea: 丢点后的特征 [1, N', F]
        aug_input_adj: 丢点后的邻接（scipy.sparse.csr_matrix）[N', N']
    """
    # 邻接转成 torch dense，便于做行列删除
    input_adj = torch.tensor(input_adj.todense().tolist())
    # 去掉 batch 维：[1,N,F] -> [N,F]
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    # 只删特征的行（对应删节点）
    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    # 邻接要删行删列
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    """子图采样增强（subgraph）。

    过程：
    - 随机选择一个中心节点
    - 迭代地从当前子图节点的邻居中扩展，直到达到目标子图大小 s_node_num
    - 其余节点全部丢弃（等效于取诱导子图）

    Args:
        input_fea: 节点特征 [1, N, F]
        input_adj: scipy.sparse 邻接 [N, N]
        drop_percent: 丢弃比例，保留节点数约为 N*(1-drop_percent)

    Returns:
        aug_input_fea: 子图特征 [1, N', F]
        aug_input_adj: 子图邻接（scipy.sparse.csr_matrix）[N', N']
    """
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))

    # 随机中心节点
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    # 逐步扩展子图节点集合
    for i in range(s_node_num - 1):
        # 收集当前子图中第 i 个节点的邻居
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        # 去重
        all_neighbor_list = list(set(all_neighbor_list))
        # 过滤掉已经在子图里的节点
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            # 没有可扩展邻居则提前结束
            break

    # 需要删除的节点 = 全体 - 子图节点
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    """从矩阵中删除指定索引的行/列。

    Args:
        input_matrix: torch Tensor（2D）
        drop_list: 需要删除的索引列表
        only_row: True 表示只删行（常用于特征矩阵）；False 表示行列都删（常用于邻接矩阵）

    Returns:
        out: 删除后的矩阵
    """
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


from utils import process


def build_aug(adj, feature, sparse, drop_percent):
    """构造两份图增强视图（主要是 edge perturbation），并生成对比学习所需标签。

    Args:
        adj: 原始邻接（scipy.sparse），形状 [N, N]
        feature: 原始特征（torch Tensor 或 numpy），形状 [N, F]
        sparse: 是否将邻接转换为 torch sparse tensor
        drop_percent: 边扰动比例

    Returns:
        features: torch.stack 后的特征集合，形状 [4, N, F]
            - 0: 原始 feature
            - 1: shuf_fts（特征行随机打乱，作为负样本）
            - 2: feature.detach()
            - 3: feature.detach()
          注：这里重复/detach 的具体用途取决于上游训练代码的取用方式。

        adjs: torch.stack 后的邻接集合，形状 [3, ...]
            - 0: 归一化后的原始邻接
            - 1: 增强视图 1（随机删/加边后再归一化）
            - 2: 增强视图 2（随机删/加边后再归一化）

        lbl: 对比学习标签，形状 [1, 2N]
            - 前 N 个为 1（正样本）
            - 后 N 个为 0（负样本）
    """
    # 两份独立的随机边增强
    aug_adj1edge = aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
    aug_adj2edge = aug_random_edge(adj, drop_percent=drop_percent)

    # 加自环并做归一化（通常是 D^{-1/2}(A+I)D^{-1/2}）
    aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
    aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    # 根据 sparse 标志把 scipy sparse 转 torch sparse 或 dense float tensor
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
        aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)
    else:
        adj = torch.FloatTensor(adj.todense())
        aug_adj1edge = torch.FloatTensor(aug_adj1edge.todense())
        aug_adj2edge = torch.FloatTensor(aug_adj2edge.todense())

    # 生成特征打乱版本：通过打乱节点顺序构造负样本
    nb_nodes = feature.shape[0]
    idx = np.random.permutation(nb_nodes)
    shuf_fts = feature[idx, :]

    # 构造 DGI 常用的二分类标签：正样本 1，负样本 0
    lbl_1 = torch.ones(1, nb_nodes)
    lbl_2 = torch.zeros(1, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), dim=1)  # shape: [1, 2N]

    # 返回 stacked 特征与 stacked 邻接，供上游一次性取多个视图/样本
    return torch.stack([feature, shuf_fts, feature.detach(), feature.detach()]), torch.stack([adj, aug_adj1edge, aug_adj2edge]), lbl
