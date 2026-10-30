"""
cgpc.py  —  Cross-Graph Contrastive (CGC) 模块

方法定位：
    L_total = L_SAMGPT + λ_cgc * L_cgc

    L_cgc : graph-level 跨图对比损失（NT-Xent / InfoNCE）

    核心设计：
    - 每张源图 → GNN + mean pooling → 图级向量 z_k
    - GRAPHCL 已有两个 augmented view，直接复用
    - 同一图的两个 view 是正对，不同图的 view 是负样本
    - 不需要 EMA teacher，不需要 codebook，无 collapse 风险
    - batch size = 源图数量（6），负样本数 = 2K-2

    下游不使用任何额外模块，零推理开销。

推荐实验：
    --use_cgpc --lambda_cgc 0.1
    消融：--lambda_cgc 0.0（纯 SAMGPT baseline）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph-level readout
# ---------------------------------------------------------------------------

def mean_pooling(h: torch.Tensor) -> torch.Tensor:
    """节点 embedding 均值池化为图级向量。

    Args:
        h : [1, N, D] 或 [N, D]

    Returns:
        z : [D]
    """
    if h.dim() == 3:
        h = h.squeeze(0)   # [N, D]
    return h.mean(dim=0)   # [D]


# ---------------------------------------------------------------------------
# NT-Xent loss（InfoNCE，对称版本）
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent loss（SimCLR 风格）。

    Args:
        z1 : [K, D]  第一组 view 的图级向量（K 张图）
        z2 : [K, D]  第二组 view 的图级向量（K 张图）
        temperature : softmax 温度

    Returns:
        loss : scalar

    正对：(z1[i], z2[i])
    负样本：同一 batch 内其他所有向量（共 2K-2 个）

    实现：把 z1 和 z2 拼成 [2K, D]，构造 2K×2K 相似度矩阵，
    正对在 off-diagonal 位置，用 cross-entropy 计算。
    """
    K = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # [2K, D]
    z = F.normalize(z, dim=-1)

    # 相似度矩阵 [2K, 2K]
    sim = z @ z.t() / temperature           # [2K, 2K]

    # 正对的 index：z1[i] 的正对是 z2[i]，即 index i+K；z2[i] 的正对是 z1[i]，即 index i
    labels = torch.cat([
        torch.arange(K, 2 * K, device=z.device),   # z1[i] → z2[i]
        torch.arange(0, K,     device=z.device),    # z2[i] → z1[i]
    ])                                               # [2K]

    # 把对角线（自身相似度）mask 掉，避免自身作为负样本
    mask = torch.eye(2 * K, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# CGC 主模块
# ---------------------------------------------------------------------------

class CGCModule(nn.Module):
    """Cross-Graph Contrastive 模块。

    对每张源图的两个 augmented view 做 graph-level 对比。
    不需要 EMA，不需要 codebook。

    Args:
        hid_dim     : GNN 隐层维度
        proj_dim    : projection head 输出维度（建议 128）
        temperature : NT-Xent 温度（建议 0.5）
    """

    def __init__(
        self,
        hid_dim:     int,
        proj_dim:    int   = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        # projection head：把图级向量映射到对比空间
        # SimCLR 论文证明加 projection head 比直接用 encoder 输出效果好
        self.projector = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, proj_dim),
        )
        self.temperature = temperature

    def forward(
        self,
        graph_embeds_view1: list,   # list of [D] tensors，每张图 view1 的图级向量
        graph_embeds_view2: list,   # list of [D] tensors，每张图 view2 的图级向量
    ):
        """计算跨图对比 loss。

        Args:
            graph_embeds_view1 : K 个 [D] tensor
            graph_embeds_view2 : K 个 [D] tensor

        Returns:
            loss     : scalar Tensor
            loss_dict: {"loss_cgc": float}
        """
        # stack 成 [K, D]
        z1 = torch.stack(graph_embeds_view1, dim=0)   # [K, D]
        z2 = torch.stack(graph_embeds_view2, dim=0)   # [K, D]

        # projection
        z1 = self.projector(z1)   # [K, proj_dim]
        z2 = self.projector(z2)   # [K, proj_dim]

        loss = nt_xent_loss(z1, z2, self.temperature)

        return loss, {"loss_cgc": float(loss.detach().cpu())}
