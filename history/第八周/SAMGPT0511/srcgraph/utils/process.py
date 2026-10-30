import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

# ------------------------------------------------------------
# 该文件是一个“杂项工具集合”，主要包含：
# 1) 从邻接中采样 1-hop/2-hop 邻居并构建子图 batch（用于子图训练/提示学习等）
# 2) t-SNE 可视化与按标签绘图
# 3) 将多个图的邻接拼成 block-diagonal 的大图（数据集融合）
# 4) DGI/GCN 相关的常用预处理：mask、特征归一化、邻接归一化、稀疏转换等
# 注：该文件部分代码来自经典实现（tkipf/gcn），并因项目需要做了改动。
# ------------------------------------------------------------


def find_2hop_neighbors_sp(adj, node):
    """在稀疏邻接（scipy sparse）上采样 1-hop 和 2-hop 邻居。

    实现细节：
    - 通过 `adj.getrow(node).todense()` 获取某节点的邻接行（dense），然后线性扫描非零项。
    - 对 1-hop 邻居数量做上限截断（最多 4 个）。
    - 对每个 1-hop 邻居，再采样最多 2 个二跳邻居。

    Args:
        adj: scipy.sparse 邻接矩阵（支持 getrow），形状 [N, N]
        node: 目标节点 id

    Returns:
        neighbors: 1-hop 邻居列表（长度 <=4）
        neighbors_2hop: 2-hop 邻居列表（长度 <= 4*2）
    """
    # print(adj.getrow(node))
    # print(adj.getrow(node).todense().A)
    nodeadj = adj.getrow(node).todense().A[0]
    neighbors = []
    # print(type(adj))
    for i in range(len(nodeadj)):
        if len(neighbors) >= 4:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if nodeadj[i] != 0 and node != i:
            neighbors.append(i)

    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        nodeadj = adj.getrow(i).todense().A[0]
        for j in range(len(nodeadj)):
            if cnt >= 2:
                break
            if nodeadj[j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop


def sp_adj(adj, node1, node2):
    """（未完成/未使用）似乎想在 COO 表示中为 (node1,node2) 定位起始位置。

    目前该函数没有返回值，也没有被调用；可能是开发过程遗留。
    """
    begin = 0
    for i in range(adj.row.shape):
        if adj.row[i] == node1:
            begin = i
            break


def find_2hop_neighbors(adj, node):
    """在 dense 邻接（如 torch/numpy 2D 数组）上采样 1-hop 和 2-hop 邻居。

    与 find_2hop_neighbors_sp 类似，但阈值不同：
    - 1-hop 最多 10 个
    - 每个 1-hop 的 2-hop 最多 4 个

    Args:
        adj: dense 邻接矩阵（支持 adj[node][i] 访问），形状 [N,N]
        node: 目标节点 id

    Returns:
        neighbors: 1-hop 邻居
        neighbors_2hop: 2-hop 邻居
    """
    neighbors = []
    # print(type(adj))
    for i in range(len(adj[node])):
        if len(neighbors) >= 10:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)

    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        for j in range(len(adj[i])):
            if cnt >= 4:
                break
            if adj[i][j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop


def build_subgraph(adj, idx_train, sparse=True):
    """为一批目标节点构建“2-hop 子图采样结果”，并打包成 PyG 风格的 batch 索引。

    输出是一个字典：
    - idx: 所有子图节点 id 的拼接（原图中的节点编号）
    - batch: 与 idx 等长，表示 idx 中每个节点属于第几个子图（类似 PyG 的 batch 向量）

    该结构常用于把多组节点集合拼成一个大 batch 进行并行处理。

    Args:
        adj: 邻接矩阵（scipy sparse 或 dense），取决于 sparse 参数
        idx_train: 目标节点索引张量，形状 [B]
        sparse: True 使用 find_2hop_neighbors_sp；False 使用 find_2hop_neighbors

    Returns:
        dict(idx=Tensor, batch=Tensor)
    """
    neighborslist = [[] for x in range(idx_train.shape[0])]
    neighbors_2hoplist = [[] for x in range(idx_train.shape[0])]
    mainindex = [[] for x in range(idx_train.shape[0])]
    mainlist = [[] for x in range(idx_train.shape[0])]

    idx_train_list = idx_train.tolist()
    for x in range(idx_train.shape[0]):
        if sparse:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors_sp(adj, idx_train[x])
        else:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors(adj, idx_train[x])

        # 子图节点集合：中心节点 + 1-hop + 2-hop
        mainlist[x] = [idx_train_list[x]] + neighborslist[x] + neighbors_2hoplist[x]
        # batch index：标识属于第 x 个子图
        mainindex[x] = [x] * len(mainlist[x])

    # 展平为一维列表
    neighborslist = sum(neighborslist, [])
    neighbors_2hoplist = sum(neighbors_2hoplist, [])
    mainlist = sum(mainlist, [])
    mainindex = sum(mainindex, [])

    return {
        'idx': torch.tensor(mainlist),
        'batch': torch.tensor(mainindex),
    }


def plotlabels(feature, Trure_labels, name):
    """对特征做 t-SNE 降维，并按类别绘制散点图保存。

    注：
    - 这里写死了只画前 4 个类别（range(4)）以及固定颜色表。
    - 保存路径固定为 'plt_graph/exceptcomputers/{name}.png'

    Args:
        feature: 特征矩阵 [N,F]（numpy array）
        Trure_labels: 标签 [N] 或 [N,1]
        name: 图标题/保存名
    """
    # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']

    S_lowDWeights = visual(feature)
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab', 'hotpink']
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(4):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker='.', c=colors[index], edgecolors=colors[index])
        plt.xticks([])
        plt.yticks([])

    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    plt.savefig('plt_graph/exceptcomputers/{}.png'.format(name), dpi=500)
    plt.show()
    plt.clf()


def visual(feat):
    """t-SNE 将高维特征降到 2 维并做 [0,1] 归一化。"""
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final


def combine_dataset(*args):
    """将多个图的邻接矩阵合并成 block-diagonal 大图（dense 路径）。

    输入多个 scipy sparse 邻接矩阵：adj1, adj2, ...
    输出：一个大的 scipy.sparse.csr_matrix，其中不同图之间无边相连。

    该实现会反复 densify + 拼接，图大时非常耗内存；更推荐用 combine_dataset_list_sp。
    """
    # print(feature1.shape)
    # print(feature2.shape)
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))

    adj = sp.csr_matrix(adj1)
    return adj


def combine_dataset_list(args):
    """combine_dataset 的 list 版本（仍走 dense 拼接）。"""
    # print(feature1.shape)
    # print(feature2.shape)
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))

    adj = sp.csr_matrix(adj1)
    return adj


def combine_dataset_list_sp(args):
    """将多个稀疏邻接拼成 block-diagonal 大图（稀疏路径，内存友好）。

    通过 scipy.sparse 的 hstack/vstack 构造：
    [A1  0 ]
    [0  A2 ]
    ...
    """
    adj1 = None

    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj
        else:
            num_rows1, num_cols1 = adj1.shape
            num_rows2, num_cols2 = adj.shape
            zeroadj1 = sp.csr_matrix((num_rows1, num_cols2))
            zeroadj2 = sp.csr_matrix((num_rows2, num_cols1))

            top = sp.hstack([adj1, zeroadj1])
            bottom = sp.hstack([zeroadj2, adj])
            adj1 = sp.vstack([top, bottom])

    return adj1.tocsr()


def parse_skipgram(fname):
    """解析 DeepWalk/node2vec 等 skip-gram 格式的 embedding 文本文件。

    文件格式约定（常见）：
    - 第一行两个数字：nb_nodes nb_features
    - 后续每行：node_id feat_1 feat_2 ...
      （此处 node_id 从 1 开始，因此读取后减 1）

    Returns:
        ret: numpy array, shape [nb_nodes, nb_features]
    """
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form
def process_tu(data, class_num):
    """将 TU 数据集中的图转换为 (features, adj) 形式。

    这里假设：
    - data.x 前 class_num 维是 one-hot 类别特征
    - 后面的维度是“raw labels”或其他属性

    Args:
        data: PyG Data
        class_num: 类别数（用于切分 data.x 的前几维）

    Returns:
        features: Tensor [N, class_num]
        adj: scipy.sparse.csr_matrix [N,N]
    """
    # print("len",nb_graphs)
    ft_size = data.num_features

    num = range(class_num)
    labelnum = range(class_num, ft_size)

    features = data.x[:, num]
    rawlabels = data.x[:, labelnum]

    e_ind = data.edge_index
    coo = sp.coo_matrix(
        (np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
        shape=(features.shape[0], features.shape[0]),
    )
    adjacency = coo
    adj = sp.csr_matrix(adjacency)

    return features, adj


def micro_f1(logits, labels):
    """计算 multi-label 任务的 micro-F1。

    说明：
    - logits 先过 Sigmoid 再 round 得到 0/1 预测
    - 然后按 tp/tn/fp/fn 计算 micro-precision/micro-recall/micro-f1
    """
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""

def adj_to_bias(adj, sizes, nhood=1):
    """将邻接矩阵转换为 GAT/DGI 常见的 attention bias（不可达位置给 -1e9）。

    给定 nhood（邻域阶数），先计算 nhood 跳可达矩阵 mt：
    - 对每个图 mt 从 I 开始
    - 重复 nhood 次：mt = mt * (A+I)
    - 把可达位置置 1，不可达置 0

    最后返回：-1e9 * (1 - mt)
      - 若 mt[i,j]=1 表示可达，bias=0
      - 若 mt[i,j]=0 表示不可达，bias=-1e9（softmax 后近似为 0）

    Args:
        adj: numpy array，形状 [G, N, N]
        sizes: 每个图实际节点数（有些实现会 padding）
        nhood: 邻域阶数

    Returns:
        bias: numpy array，形状同 adj
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################


def parse_index_file(filename):
    """读取 index 文件（每行一个整数）。"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """根据 idx 生成长度为 l 的 0/1 mask。"""
    mask = np.zeros(l)
    mask[idx] = 1
    # 这里使用 np.bool（已弃用），保持原实现不改动
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """加载 tkipf/gcn 版本的 Planetoid 数据（非 PyG）。

    从 data/ind.{dataset}.{x,y,tx,ty,allx,ally,graph} 读取，并构造：
    - adj: scipy sparse adjacency
    - features: scipy sparse features
    - labels: numpy array
    - idx_train/idx_val/idx_test: 索引

    注意：本实现使用了相对路径 "data/"，运行时工作目录需正确。
    """
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """将 scipy sparse 转为 (coords, values, shape) 的 tuple 表示。

    Args:
        sparse_mx: scipy sparse 或 list[scipy sparse]
        insert_batch: 是否额外插入 batch 维（coords 变成 [nnz,3]，shape 前面加 1）

    Returns:
        (coords, values, shape) 或对应 list
    """

    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """特征标准化：按训练集 mask 计算均值/方差并做 z-score。

    Args:
        f: scipy sparse features
        train_mask: numpy bool mask

    Returns:
        f: standardized dense matrix
    """
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """对特征做行归一化（每行除以行和）。

    Returns:
        dense_features: features.todense()
        tuple_features: sparse_to_tuple(features)
    """
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """对邻接做对称归一化：D^{-1/2} A D^{-1/2}。"""
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """GCN 常用的邻接预处理：先加自环再对称归一化，然后转 tuple。"""
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy sparse COO/CSR 转为 torch.sparse.FloatTensor。"""
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




