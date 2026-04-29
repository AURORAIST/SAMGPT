import torch
import torch.nn as nn


# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    """平均读出（readout）层：把节点表示聚合成图级表示。

    常用于图对比学习 / DGI 这类方法中，将节点特征 `seq` 沿节点维做平均：
    - 无 mask：对所有节点直接平均
    - 有 mask：只对 mask=1 的节点做加权平均（等价于对被选中节点的平均）

    Shapes:
        seq: [B, N, F]
        msk: [B, N] 或 None
        return: [B, F]
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        """计算图级向量。

        Args:
            seq: 节点表示序列，形状 [batch, nodes, features]
            msk: 节点掩码，形状 [batch, nodes]；为 None 表示不使用掩码

        Returns:
            图级表示向量，形状 [batch, features]
        """
        if msk is None:
            # 沿节点维（dim=1）做均值
            return torch.mean(seq, 1)
        else:
            # 扩展 mask 以便与 seq 在特征维上广播相乘
            # msk: [B,N] -> [B,N,1]
            msk = torch.unsqueeze(msk, -1)
            # masked average: sum(seq*msk)/sum(msk)
            # 其中 sum(msk) 是每个样本被选中节点的数量（广播到特征维）
            return torch.sum(seq * msk, 1) / torch.sum(msk)

