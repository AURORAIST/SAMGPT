import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch_sparse import spmm


class GAT(nn.Module):
    """基于 PyG `GATConv` 的图注意力层封装。

    该模块做的事情很简单：
    1) 使用 `GATConv` 从输入节点特征 `x` 和图结构 `adj` 计算新的节点表示
    2) 经过 PReLU 激活
    3) 施加 dropout

    注意：
    - PyG 的 `GATConv` 期望图结构通常为 `edge_index`（COO 格式，形状 [2, E]），
      但也可能支持 `SparseTensor` 等形式（取决于版本）。
    - 本文件虽然导入了 `dense_to_sparse` / `spmm`，但当前实现并未使用它们；
      这通常意味着上游已经把邻接矩阵处理成 `GATConv` 可接受的格式。
    """

    def __init__(self, in_ft, out_ft, nheads=2, concat=True, dropout=0.6, alpha=0.2, bias=True):
        """初始化 GAT 层。

        Args:
            in_ft: 输入特征维度。
            out_ft: 每个 head 的输出维度（当 concat=True 时，最终维度为 out_ft * nheads）。
            nheads: 注意力头数。
            concat: 是否拼接多头输出；若为 False 则做平均聚合。
            dropout: dropout 概率（同时传给 GATConv 和本层的 Dropout）。
            alpha: 通常表示 LeakyReLU 负斜率（PyG 的 GATConv 参数名一般为 negative_slope）。
                   这里保留该参数但未实际传入 `GATConv`，因此当前实现中它不起作用。
            bias: 是否使用 bias。
        """
        super(GAT, self).__init__()
        # PyG 原生 GATConv：内部包含注意力系数计算与邻居聚合。
        self.gat = GATConv(in_ft, out_ft, heads=nheads, concat=concat, dropout=dropout, bias=bias)
        # 论文/实践中常用的激活函数，这里选择 PReLU。
        self.act = nn.PReLU()
        # 输出端再做一次 dropout（与 GATConv 内部的 dropout 不同位置）。
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """前向。

        Args:
            input: 二元组/列表 (x, adj)
                - x: 节点特征，形状通常为 [N, in_ft]
                - adj: 图结构，通常为 PyG 的 edge_index（[2, E]）或等价稀疏表示

        Returns:
            更新后的节点特征，形状取决于 `concat`：
                - concat=True: [N, out_ft * nheads]
                - concat=False: [N, out_ft]
        """
        # 约定：input[0] 是节点特征，input[1] 是图结构
        x = input[0]
        adj = input[1]

        # 顺序：GATConv -> 激活 -> dropout
        return self.dropout(self.act(self.gat(x, adj)))
