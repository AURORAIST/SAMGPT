"""
cgpc.py  —  Predictive-Codebook SAMGPT (CGPC) 模块

方法定位：
    L_total = L_SAMGPT + λ_ctx * L_ctx + λ_code * L_code

    L_ctx  : student(masked view, prompt-enhanced) -> teacher(clean view, prompt-enhanced)
    L_code : KL( Q_teacher || P_student ) on shared codebook

    关键设计：
    - student 看 feature-masked 图，teacher 看 clean 图（真正的 predictive coding）
    - teacher/student 都走 prompt-enhanced 路径，与 SAMGPT 主路径一致
    - teacher prompts 也做 EMA 更新（完整 teacher）
    - 所有源图共享同一个 learnable codebook（跨图共享 latent space）
    - 下游不使用 teacher / predictor / codebook，零额外推理开销

推荐实验顺序：
    Stage 1: --lambda_context 0.05 --lambda_code 0.0  --mask_type element
    Stage 2: --lambda_context 0.05 --lambda_code 0.05 --mask_type element
    Ablation: --mask_rate 0.0  (no-mask control)
              --mask_type node (stronger mask)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """两层 MLP：Linear -> LayerNorm -> PReLU -> Linear。

    LayerNorm 而非 BatchNorm：图任务 batch size 不固定，LN 对任意大小稳定。
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
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature masking（构造 student masked view）
# ---------------------------------------------------------------------------

def feature_dropout(
    x: torch.Tensor,
    drop_rate: float = 0.2,
    mask_type: str = "element",
) -> torch.Tensor:
    """特征 mask，支持两种粒度。

    mask_type="element"（推荐第一轮）：
        元素级 dropout，每个特征值独立以 drop_rate 概率置零。
        更温和，适合稀疏/稠密特征。

    mask_type="node"：
        节点级 dropout，整行置零。
        更强的扰动，适合作为消融对比。

    Args:
        x        : [N, F] 或 [1, N, F]（自动处理 batch 维）
        drop_rate: mask 概率
        mask_type: "element" 或 "node"

    Returns:
        x_masked : 与输入同 shape
    """
    if drop_rate <= 0.0:
        return x

    squeeze = False
    if x.dim() == 3:
        x = x.squeeze(0)
        squeeze = True

    assert x.dim() == 2, f"[feature_dropout] expected [N,F] after squeeze, got {x.shape}"

    if mask_type == "node":
        # 节点粒度：整行置零
        mask = (torch.rand(x.shape[0], device=x.device) > drop_rate).float().unsqueeze(1)
    else:
        # 元素粒度：每个值独立 mask
        mask = (torch.rand_like(x) > drop_rate).float()

    x_masked = x * mask

    if squeeze:
        x_masked = x_masked.unsqueeze(0)
    return x_masked


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def cosine_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """2 - 2 * cosine_similarity，均值。target 应已 detach。"""
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Code usage 统计（监控 codebook collapse）
# ---------------------------------------------------------------------------

@torch.no_grad()
def code_usage_stats(q: torch.Tensor) -> dict:
    """计算 codebook 使用统计，用于监控 collapse。

    Args:
        q: [N, M] teacher soft assignment（已 softmax）

    Returns:
        dict with:
            code_entropy  : 平均 code 使用的 entropy（越高越均匀）
            code_active   : 有效 code 数（usage > 1/(10*M) 的 code 数量）
            code_max_usage: 最大单 code 使用率
    """
    usage   = q.mean(dim=0)                                    # [M]
    entropy = -(usage * (usage + 1e-12).log()).sum().item()
    active  = (usage > 1.0 / (10.0 * usage.numel())).float().sum().item()
    max_use = usage.max().item()
    return {
        "code_entropy":   entropy,
        "code_active":    active,
        "code_max_usage": max_use,
    }


# ---------------------------------------------------------------------------
# CGPC 主模块
# ---------------------------------------------------------------------------

class CGPCModule(nn.Module):
    """Predictive-Codebook 模块。

    包含：
    - predictor : student embedding -> prediction space（与 teacher 对齐）
    - codebook  : M × D 的共享 learnable codes（所有源图共享）

    forward 流程：
    1. pred = predictor(h_student)                          # 只 forward 一次
    2. L_ctx  = cosine_loss(pred, h_teacher)
    3. q = softmax(h_teacher @ C^T / τ_t)                  # teacher code dist（no grad）
    4. log_p = log_softmax(pred @ C^T / τ_s)               # student code dist
    5. L_code = KL(q || p)

    Args:
        hid_dim      : GNN 隐层维度
        num_codes    : codebook 大小 M，建议 256
        tau_teacher  : teacher softmax 温度（较小 → 更尖锐）
        tau_student  : student softmax 温度（略大于 teacher）
        sample_size  : CGPC loss 的节点采样数（0 = 不采样，全图）
                       大图时建议设 4096 或 8192，避免 N×M 显存爆炸
    """

    def __init__(
        self,
        hid_dim:            int,
        num_codes:          int   = 256,
        tau_teacher:        float = 0.07,
        tau_student:        float = 0.10,
        sample_size:        int   = 0,
        uniform_reg_weight: float = 5.0,   # codebook 正交正则权重，防 collapse
    ):
        super().__init__()
        self.predictor          = Predictor(hid_dim, hid_dim, hid_dim)
        self.codebook           = nn.Parameter(torch.empty(num_codes, hid_dim))
        nn.init.xavier_uniform_(self.codebook)
        self.tau_teacher        = tau_teacher
        self.tau_student        = tau_student
        self.sample_size        = sample_size
        self.uniform_reg_weight = uniform_reg_weight
        # encourage student distribution uniformity (grad flows through student)
        self.usage_uniform_weight = 1.0

    def _maybe_sample(
        self,
        h_stu: torch.Tensor,
        h_tea: torch.Tensor,
    ):
        """如果节点数超过 sample_size，随机采样子集。"""
        N = h_stu.size(0)
        if self.sample_size > 0 and N > self.sample_size:
            idx   = torch.randperm(N, device=h_stu.device)[:self.sample_size]
            return h_stu[idx], h_tea[idx]
        return h_stu, h_tea

    def _code_loss(
        self,
        pred:      torch.Tensor,   # [N, D]  predictor 输出
        h_teacher: torch.Tensor,   # [N, D]  已 detach
    ):
        """KL( Q_teacher || P_student ) + codebook 正交正则。

        防 collapse 策略：codebook orthogonality regularization。
        让不同 code 之间尽量正交，防止退化成同一方向。

        L_code = KL(Q_t || P_s) + uniform_reg_weight * L_orth
        L_orth = mean((C C^T - I)^2)

        与 uniform prior 的区别：
        - uniform prior 约束的是 q（在 no_grad 里，梯度无法传回）
        - orthogonality reg 直接约束 codebook 参数，梯度路径清晰
        """
        # keep codebook vectors normalized to stabilize training
        with torch.no_grad():
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)
        code     = F.normalize(self.codebook, dim=-1)   # [M, D]

        with torch.no_grad():
            h_t = F.normalize(h_teacher, dim=-1)
            q   = torch.softmax(h_t @ code.t() / self.tau_teacher, dim=-1)  # [N, M]
            stats = code_usage_stats(q)

        h_s   = F.normalize(pred, dim=-1)
        log_p = torch.log_softmax(h_s @ code.t() / self.tau_student, dim=-1)

        # 主 loss：student 预测 teacher assignment
        loss_kl = F.kl_div(log_p, q, reduction="batchmean")

        # 使用均匀性正则（作用于 student 的平均分配，梯度可回传）
        if self.usage_uniform_weight is not None and self.usage_uniform_weight > 0:
            p = torch.softmax(h_s @ code.t() / self.tau_student, dim=-1)  # [N, M]
            usage_p = p.mean(dim=0)  # [M]
            # 负熵（= sum p log p），最小化可促使分布更平坦（即最大化熵）
            loss_usage_uniform = (usage_p * (usage_p + 1e-12).log()).sum()
        else:
            loss_usage_uniform = torch.tensor(0.0, device=pred.device)

        # codebook 正交正则：防止不同 code 退化成同一方向
        # code_norm @ code_norm.T 的非对角元素应接近 0
        sim   = code @ code.t()                                          # [M, M]
        eye   = torch.eye(sim.size(0), device=sim.device)
        loss_orth = (sim - eye).pow(2).mean()

        loss = loss_kl + self.uniform_reg_weight * loss_orth + self.usage_uniform_weight * loss_usage_uniform

        # 若检测到坍缩（有效 code 数极少），标记在训练循环中重置，避免 forward 内原地修改参数
        # 把阈值暴露为 heuristics：当 active < 1% 的 code 数时认为坍缩
        try:
            M = code.size(0)
            active = stats.get("code_active", 0)
            # debug log for collapse detection
            if active is None:
                print('[CGPC] code_active not found in stats')
            else:
                print(f'[CGPC] code usage active={active}/{M}, max_usage={stats.get("code_max_usage",0):.4f}, entropy={stats.get("code_entropy",0):.4f}')
            if active < max(1, 0.01 * M):
                print('[CGPC] codebook usage low -> flag reset after step')
                setattr(self, "_needs_codebook_reset", True)
                stats["code_reset"] = 1
        except Exception as e:
            print('[CGPC] code usage stats check failed:', e)
            pass

        return loss, stats

    def forward(
        self,
        h_student:      torch.Tensor,   # [N, D]  student（masked view，prompt-enhanced）
        h_teacher:      torch.Tensor,   # [N, D]  teacher（clean view，prompt-enhanced，已 detach）
        lambda_context: float = 0.05,
        lambda_code:    float = 0.0,
    ):
        """计算 CGPC loss。

        Returns:
            loss_cgpc : scalar Tensor（保留梯度）
            loss_dict : 各分项 float + code usage stats（已 detach，用于日志）
        """
        dev = h_student.device

        loss_ctx  = torch.tensor(0.0, device=dev)
        loss_code = torch.tensor(0.0, device=dev)
        code_stats = {}

        # 节点采样（大图降低显存）
        h_stu_c, h_tea_c = self._maybe_sample(h_student, h_teacher)

        # predictor 只 forward 一次，context 和 code 共用
        pred = None
        if lambda_context > 0 or lambda_code > 0:
            pred = self.predictor(h_stu_c)

        if lambda_context > 0:
            loss_ctx = cosine_prediction_loss(pred, h_tea_c)

        if lambda_code > 0:
            loss_code, code_stats = self._code_loss(pred, h_tea_c)

        loss_cgpc = lambda_context * loss_ctx + lambda_code * loss_code

        loss_dict = {
            "loss_context":   float(loss_ctx.detach().cpu()),
            "loss_code":      float(loss_code.detach().cpu()),
            "loss_cgpc":      float(loss_cgpc.detach().cpu()),
            # code usage（lambda_code=0 时为空 dict，不影响日志）
            **{k: float(v) for k, v in code_stats.items()},
        }
        return loss_cgpc, loss_dict


# ---------------------------------------------------------------------------
# Teacher 构建 & EMA 更新
# ---------------------------------------------------------------------------

def build_ema_teacher(student: nn.Module) -> nn.Module:
    """从 student module 构建 EMA teacher。

    deepcopy + eval() + requires_grad=False。
    可以传 GcnLayers、nn.ModuleList（prompts）等任意 nn.Module。
    """
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float = 0.99):
    """EMA 更新：teacher = momentum * teacher + (1 - momentum) * student。

    含参数数量和 shape 双重断言。
    调用时机：optimizer.step() 之后。
    """
    t_params = list(teacher.parameters())
    s_params = list(student.parameters())

    assert len(t_params) == len(s_params), (
        f"[ema_update] parameter count mismatch: "
        f"teacher={len(t_params)}, student={len(s_params)}."
    )
    for p_t, p_s in zip(t_params, s_params):
        assert p_t.shape == p_s.shape, (
            f"[ema_update] shape mismatch: "
            f"teacher {tuple(p_t.shape)} vs student {tuple(p_s.shape)}."
        )
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def _kmeans_init(points: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """Simple k-means in PyTorch for moderate-sized pretraining (cpu/gpu)."""
    device = points.device
    N, D = points.shape
    # random init: pick k unique points
    if N <= k:
        centroids = points[:k].clone()
        if centroids.size(0) < k:
            # pad by repeating
            reps = (k - centroids.size(0))
            centroids = torch.cat([centroids, centroids[:reps]], dim=0)
        return centroids

    # choose k random seeds
    ids = torch.randperm(N, device=device)[:k]
    centroids = points[ids].clone()

    for _ in range(iters):
        # assign
        dist = torch.cdist(points, centroids, p=2)  # [N, k]
        idx = dist.argmin(dim=1)
        # update
        new_centroids = []
        for j in range(k):
            sel = points[idx == j]
            if sel.numel() == 0:
                # reinit empty centroid
                new_centroids.append(centroids[j].unsqueeze(0))
            else:
                new_centroids.append(sel.mean(dim=0, keepdim=True))
        centroids = torch.cat(new_centroids, dim=0)

    return centroids


def _attach_cgpc_utils():
    def reset_codebook(self, mode: str = "xavier"):
        with torch.no_grad():
            if mode == "xavier":
                nn.init.xavier_uniform_(self.codebook)
            else:
                nn.init.normal_(self.codebook, std=0.02)
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)
            print('[CGPC] reset_codebook executed, mode=', mode)

    def normalize_codebook(self):
        with torch.no_grad():
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)
        print('[CGPC] normalize_codebook executed')

    def pretrain_codebook(self, embeddings: torch.Tensor, iters: int = 20):
        # embeddings: [N, D]
        with torch.no_grad():
            centroids = _kmeans_init(embeddings, self.codebook.size(0), iters=iters)
            if centroids.shape[1] != self.codebook.shape[1]:
                raise ValueError("embedding dim mismatch for codebook pretrain")
            self.codebook.data.copy_(centroids.to(self.codebook.device))
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)
        print(f'[CGPC] pretrain_codebook finished: embeddings={embeddings.shape}, codes={self.codebook.shape}, iters={iters}')

    setattr(CGPCModule, "reset_codebook", reset_codebook)
    setattr(CGPCModule, "normalize_codebook", normalize_codebook)
    setattr(CGPCModule, "pretrain_codebook", pretrain_codebook)


_attach_cgpc_utils()
