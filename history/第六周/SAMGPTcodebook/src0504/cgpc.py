"""
cgpc.py  —  Cross-Graph Predictive Coding (CGPC) 模块

包含：
  - Predictor              : 两层 MLP（LayerNorm 版，对图任务更稳定）
  - CGPCModule             : 封装三个预测头 + loss 计算
  - cosine_prediction_loss : 2 - 2·cosine，用于 context 预测
  - build_ema_teacher      : 从 student encoder 构建 teacher（deepcopy + 冻结）
  - ema_update             : EMA 更新 teacher encoder 参数
  - compute_struct_targets : 预计算结构 target（可按需开关各项）

推荐第一轮实验配置：
    lambda_context = 0.05 ~ 0.1
    lambda_feat    = 0.0
    lambda_struct  = 0.0
    use_rwse       = False
    use_kcore      = False
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Predictor MLP（LayerNorm 版）
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """两层 MLP：Linear -> LayerNorm -> PReLU -> Linear。

    使用 LayerNorm 而非 BatchNorm1d，原因：
    - 图任务每次 forward 是整图所有节点，batch size 不固定；
    - graph-level representation 可能是 [1, D]，BN 会报错或统计不稳定；
    - LN 对任意 batch size 均稳定。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]  ->  [N, out_dim]
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def cosine_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """2 - 2 * cosine_similarity，均值。

    pred / target: [N, D]
    target 应已 stop-gradient（.detach()）。
    """
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# CGPC 主模块
# ---------------------------------------------------------------------------

class CGPCModule(nn.Module):
    """封装三个预测头及其 loss 计算。

    Args:
        hid_dim    : GNN 隐层维度（student/teacher 输出维度）
        feat_dim   : 对齐后的节点特征维度（unify_dim）
        struct_dim : 结构 target 维度（由 compute_struct_targets 决定）
    """

    def __init__(self, hid_dim: int, feat_dim: int, struct_dim: int):
        super().__init__()
        self.pred_context = Predictor(hid_dim, hid_dim, hid_dim)
        self.pred_feat    = Predictor(hid_dim, hid_dim, feat_dim)
        self.pred_struct  = Predictor(hid_dim, hid_dim, struct_dim)

    def forward(
        self,
        h_student: torch.Tensor,   # [N, D]  student node embeddings
        h_teacher: torch.Tensor,   # [N, D]  teacher node embeddings（外部已 detach）
        y_feat:    torch.Tensor,   # [N, feat_dim]
        y_struct:  torch.Tensor,   # [N, struct_dim]
        lambda_context: float = 0.1,
        lambda_feat:    float = 0.0,
        lambda_struct:  float = 0.0,
    ):
        """计算三个 CGPC loss 并加权求和。

        Returns:
            loss_cgpc : 加权总 loss（scalar Tensor，保留梯度）
            loss_dict : 各分项的 float 值（用于日志，已 detach）
        """
        dev = h_student.device

        loss_context = torch.tensor(0.0, device=dev)
        loss_feat    = torch.tensor(0.0, device=dev)
        loss_struct  = torch.tensor(0.0, device=dev)

        if lambda_context > 0:
            pred_ctx     = self.pred_context(h_student)
            loss_context = cosine_prediction_loss(pred_ctx, h_teacher)

        if lambda_feat > 0:
            pred_feat = self.pred_feat(h_student)
            loss_feat = F.mse_loss(pred_feat, y_feat)

        if lambda_struct > 0:
            pred_struct = self.pred_struct(h_student)
            loss_struct = F.mse_loss(pred_struct, y_struct)

        loss_cgpc = (
            lambda_context * loss_context
            + lambda_feat   * loss_feat
            + lambda_struct * loss_struct
        )

        loss_dict = {
            "loss_context": float(loss_context.detach().cpu()),
            "loss_feat":    float(loss_feat.detach().cpu()),
            "loss_struct":  float(loss_struct.detach().cpu()),
            "loss_cgpc":    float(loss_cgpc.detach().cpu()),
        }
        return loss_cgpc, loss_dict


# ---------------------------------------------------------------------------
# Teacher 构建 & EMA 更新
# ---------------------------------------------------------------------------

def build_ema_teacher(student: nn.Module) -> nn.Module:
    """从 student encoder 构建 EMA teacher。

    - deepcopy 保证结构完全一致；
    - eval() 关闭 dropout / BN 的训练行为；
    - requires_grad=False 确保不参与反传。

    用法：
        teacher = build_ema_teacher(model.gcn)
    """
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float = 0.99):
    """用 student 参数 EMA 更新 teacher。

    teacher_param = momentum * teacher_param + (1 - momentum) * student_param

    会在调用前检查参数数量是否一致，避免传错模块。
    """
    t_params = list(teacher.parameters())
    s_params = list(student.parameters())
    assert len(t_params) == len(s_params), (
        f"[ema_update] teacher ({len(t_params)}) and student ({len(s_params)}) "
        "parameter counts differ — make sure you pass matching modules."
    )
    for p_t, p_s in zip(t_params, s_params):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


# ---------------------------------------------------------------------------
# 结构 target 预计算
# ---------------------------------------------------------------------------

def _compute_clustering(adj_sp: sp.spmatrix, num_nodes: int) -> np.ndarray:
    """计算每个节点的聚类系数。

    正确公式：
        clust_v = triangles(v) / (d_v * (d_v - 1))
        triangles(v) = (A^3)_{vv}   （三阶闭合路径数）

    注意：(A^2)_{vv} = d_v（对无自环图），不是三角形数，不能用。

    Args:
        adj_sp   : scipy sparse 邻接矩阵
        num_nodes: 节点数

    Returns:
        clust: np.ndarray [num_nodes]，值域 [0, 1]
    """
    # 保证对称、无自环、0/1 邻接
    A = adj_sp.tocsr().astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.array(A.sum(axis=1)).flatten()

    # (A^3)_{vv} = 每个节点参与的三角形数 × 2
    A2   = A.dot(A)
    tri2 = np.array(A2.dot(A).diagonal()).flatten()   # diag(A^3)

    clust = np.zeros(num_nodes, dtype=np.float32)
    mask  = deg >= 2
    clust[mask] = (tri2[mask] / (deg[mask] * (deg[mask] - 1.0))).astype(np.float32)
    return clust


def _compute_kcore_nx(adj_sp: sp.spmatrix, num_nodes: int) -> np.ndarray:
    """用 networkx 计算 k-core 编号（比纯 Python 循环快得多）。

    如果 networkx 不可用，返回全零数组（相当于不使用 k-core）。
    """
    try:
        import networkx as nx
    except ImportError:
        return np.zeros(num_nodes, dtype=np.float32)

    A = adj_sp.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A = ((A + A.T) > 0).astype(np.float32).tocsr()

    G = nx.from_scipy_sparse_array(A)
    core_dict = nx.core_number(G)

    core = np.array([core_dict.get(i, 0) for i in range(num_nodes)], dtype=np.float32)
    return core


def _compute_rwse(adj_sp: sp.spmatrix, num_nodes: int, steps: int) -> np.ndarray:
    """计算随机游走结构编码 RWSE。

    RWSE_v^(r) = (P^r)_{vv}，其中 P = D^{-1} A。

    注意：对大图（>1 万节点）开销较高，建议第一轮实验关闭（use_rwse=False）。

    Returns:
        rwse: np.ndarray [num_nodes, steps]
    """
    A = adj_sp.tocsr().astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()

    deg     = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    P       = sp.diags(deg_inv).dot(A)   # 行归一化转移矩阵

    rwse = np.zeros((num_nodes, steps), dtype=np.float32)
    P_r  = sp.eye(num_nodes, format='csr')

    for r in range(steps):
        P_r        = P_r.dot(P)
        rwse[:, r] = np.array(P_r.diagonal()).flatten().astype(np.float32)

    return rwse


def compute_struct_targets(
    adj_sp:    sp.spmatrix,
    num_nodes: int,
    rw_steps:  int  = 8,
    use_rwse:  bool = False,
    use_kcore: bool = False,
    use_clust: bool = True,
) -> torch.Tensor:
    """预计算结构 target，返回图内标准化后的 FloatTensor。

    各列含义（按顺序）：
      0 : log(1 + degree)
      1 : rank-normalized degree
      2 : clustering coefficient          （use_clust=True 时）
      3 : k-core（图内归一化）             （use_kcore=True 时）
      4…: RWSE r=1…rw_steps               （use_rwse=True 时）

    推荐第一轮：use_clust=False, use_kcore=False, use_rwse=False
    → struct_dim = 2，最轻量，先验证 context loss 有效再逐步加。

    Args:
        adj_sp   : scipy sparse 邻接矩阵（原始，未加自环，未归一化）
        num_nodes: 节点数
        rw_steps : RWSE 步数（use_rwse=True 时生效）
        use_rwse : 是否计算 RWSE
        use_kcore: 是否计算 k-core（需要 networkx 或较慢的纯 Python）
        use_clust: 是否计算 clustering coefficient

    Returns:
        y_struct: torch.FloatTensor [num_nodes, struct_dim]，已做图内标准化
    """
    A = adj_sp.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.array(A.sum(axis=1)).flatten().astype(np.float32)

    # 1. log degree
    log_deg = np.log1p(deg).reshape(-1, 1)

    # 2. rank-normalized degree
    sorted_idx      = np.argsort(deg)
    rank            = np.zeros(num_nodes, dtype=np.float32)
    rank[sorted_idx] = np.arange(num_nodes, dtype=np.float32)
    rank            = (rank / max(num_nodes - 1, 1)).reshape(-1, 1)

    parts = [log_deg, rank]

    # 3. clustering coefficient（可选）
    if use_clust:
        clust = _compute_clustering(A, num_nodes).reshape(-1, 1)
        parts.append(clust)

    # 4. k-core（可选，依赖 networkx）
    if use_kcore:
        core     = _compute_kcore_nx(A, num_nodes)
        core_max = core.max()
        core     = (core / (core_max + 1e-9)).reshape(-1, 1)
        parts.append(core)

    # 5. RWSE（可选，大图慎用）
    if use_rwse and rw_steps > 0:
        rwse = _compute_rwse(A, num_nodes, rw_steps)   # [N, rw_steps]
        parts.append(rwse)

    y_struct = np.concatenate(parts, axis=1).astype(np.float32)   # [N, struct_dim]

    # 图内标准化（每列独立）
    mean     = y_struct.mean(axis=0, keepdims=True)
    std      = y_struct.std(axis=0,  keepdims=True) + 1e-9
    y_struct = (y_struct - mean) / std

    return torch.FloatTensor(y_struct)


def struct_dim_from_args(rw_steps: int, use_rwse: bool, use_kcore: bool, use_clust: bool) -> int:
    """根据开关计算 struct_dim，与 compute_struct_targets 保持一致。

    用于在模型初始化时确定 Predictor 的输出维度。
    """
    dim = 2                          # log_deg + rank
    if use_clust: dim += 1
    if use_kcore: dim += 1
    if use_rwse:  dim += rw_steps
    return dim
