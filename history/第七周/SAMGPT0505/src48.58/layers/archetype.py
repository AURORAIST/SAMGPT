"""
archetype.py — 域感知原型分析模块（Domain-Aware Archetypal Analysis, DA-AA）

理论基础：
  原型分析（AA）强制原型为数据凸包的顶点，天然防坍缩且梯度无偏。
  域感知扩展通过共享 + 域特化原型拆分，解决多域凸包主导问题，保证小域表达。

核心设计：
  - Z_shared ∈ R^{K_s × D}：所有域共享原型，由所有域梯度共同优化
  - Z_dom   ∈ R^{M × K_d × D}：域特化原型，每域独立
  - 前向：x_out = x + (x̂ - x)，残差形式，保证信息不丢失
  - 防坍缩：顶点损失 L_vert + 温度缩放 softmax 系数

用法：
  aa = DomainAwareAA(n_domains=4, n_shared=8, n_domain=4, dim=256)
  x_out, losses = aa(x, domain_id=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 单纯形投影（保证系数非负且和为1）
# ---------------------------------------------------------------------------

def project_simplex(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """将向量 v 投影到概率单纯形（非负且和为1）。

    使用排序算法，O(K log K)，支持批量操作。

    Args:
        v : [..., K]  任意实数向量
        eps : 数值稳定性小量

    Returns:
        w : [..., K]  投影后的概率向量
    """
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1)
    rho_candidates = u - (cssv - 1.0) / (torch.arange(1, v.shape[-1] + 1,
                                                        device=v.device,
                                                        dtype=v.dtype))
    mask = rho_candidates > 0
    # 找最后一个满足条件的位置
    rho_idx = mask.long().flip(-1).argmax(dim=-1, keepdim=True)
    rho_idx = v.shape[-1] - 1 - rho_idx
    theta = (cssv.gather(-1, rho_idx) - 1.0) / (rho_idx.float() + 1.0)
    w = (v - theta).clamp(min=0.0)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    return w


# ---------------------------------------------------------------------------
# 软最小距离（可微顶点损失）
# ---------------------------------------------------------------------------

def soft_min_distance(z: torch.Tensor, x: torch.Tensor,
                      temperature: float = 0.1) -> torch.Tensor:
    """计算每个原型到最近数据点的软最小距离（可微）。

    L_vert = Σ_k softmin_i ||z_k - x_i||²

    Args:
        z : [K, D]  原型
        x : [N, D]  数据点
        temperature : softmin 温度（越小越接近硬最小）

    Returns:
        loss : scalar
    """
    # [K, N]
    dist = torch.cdist(z, x, p=2).pow(2)  # [K, N]
    # softmin：-temperature * log(Σ exp(-d/temperature))
    soft_min = -temperature * torch.logsumexp(-dist / (temperature + 1e-8), dim=-1)  # [K]
    return soft_min.mean()


# ---------------------------------------------------------------------------
# HSIC（希尔伯特-施密特独立准则）
# ---------------------------------------------------------------------------

def hsic_normalized(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """计算归一化 HSIC（结构-语义解耦损失）。

    使用 RBF 核，O(N²) 复杂度，适合中等规模节点数。

    Args:
        X : [N, D1]
        Y : [N, D2]
        sigma : RBF 核带宽

    Returns:
        hsic : scalar（越小越独立）
    """
    N = X.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=X.device)

    # 中心化核矩阵
    def rbf_kernel(A):
        dist = torch.cdist(A, A, p=2).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))

    def center_kernel(K):
        H = torch.eye(N, device=K.device) - 1.0 / N
        return H @ K @ H

    Kx = center_kernel(rbf_kernel(X))
    Ky = center_kernel(rbf_kernel(Y))

    hsic_val = (Kx * Ky).sum() / ((N - 1) ** 2 + 1e-8)
    # 归一化
    norm_x = ((Kx * Kx).sum() / ((N - 1) ** 2 + 1e-8)).sqrt().clamp(min=1e-8)
    norm_y = ((Ky * Ky).sum() / ((N - 1) ** 2 + 1e-8)).sqrt().clamp(min=1e-8)
    return hsic_val / (norm_x * norm_y)


# ---------------------------------------------------------------------------
# 域感知原型分析主模块
# ---------------------------------------------------------------------------

class DomainAwareAA(nn.Module):
    """域感知原型分析模块（DA-AA）。

    Args:
        n_domains   : 域（数据集）数量 M
        n_shared    : 共享原型数量 K_s
        n_domain    : 每域特化原型数量 K_d（初始值，支持动态扩展）
        dim         : 嵌入维度 D
        temperature : softmax 温度（默认 0.1/√D）
        lambda_vert : 顶点损失系数
        lambda_res  : 残差幅度惩罚系数
        lambda_hsic : HSIC 解耦损失系数（0 表示禁用）
        lambda_gc   : 梯度一致性损失系数
        max_k_domain: 每域最大原型数（动态扩展上限）
        use_sparse_coeff : 是否对系数做 Top-k 稀疏投影
        sparse_topk : 稀疏投影保留的 Top-k 个系数
    """

    def __init__(
        self,
        n_domains: int,
        n_shared: int = 8,
        n_domain: int = 4,
        dim: int = 256,
        temperature: float = None,
        lambda_vert: float = 0.05,
        lambda_res: float = 0.01,
        lambda_hsic: float = 0.0,
        lambda_gc: float = 0.1,
        max_k_domain: int = 12,
        use_sparse_coeff: bool = False,
        sparse_topk: int = 3,
    ):
        super(DomainAwareAA, self).__init__()

        self.n_domains = n_domains
        self.n_shared = n_shared
        self.dim = dim
        self.lambda_vert = lambda_vert
        self.lambda_res = lambda_res
        self.lambda_hsic = lambda_hsic
        self.lambda_gc = lambda_gc
        self.max_k_domain = max_k_domain
        self.use_sparse_coeff = use_sparse_coeff
        self.sparse_topk = sparse_topk

        # 温度：用可学习标量，初始值 1.0（对应归一化后的距离空间）
        # 不再用 0.1/√D，那个值对256维来说太小（≈0.006），导致softmax极度尖锐
        init_temp = temperature if temperature is not None else 1.0
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))

        # 共享原型：[K_s, D]
        self.z_shared = nn.Parameter(torch.empty(n_shared, dim))

        # 域特化原型：每域独立，用 ParameterList 存储（支持动态扩展）
        # 初始每域 n_domain 个原型
        self._k_domain = [n_domain] * n_domains  # 当前每域原型数（可变）
        self.z_domain = nn.ParameterList([
            nn.Parameter(torch.empty(n_domain, dim))
            for _ in range(n_domains)
        ])

        # HSIC 熔断状态
        self._hsic_enabled = (lambda_hsic > 0.0)
        self._hsic_fuse_count = 0
        self._hsic_fuse_window = 5  # 连续熔断次数阈值

        # 动态管理统计（每域原型使用率的 EMA）
        self._usage_ema = [
            torch.zeros(n_domain)
            for _ in range(n_domains)
        ]
        self._ema_decay = 0.99
        self._epoch_counter = 0

        # 梯度一致性：存储各域对共享原型的梯度（用于 loss 计算）
        self._shared_grads = []  # list of [K_s, D] tensors

        self._init_parameters()

    @property
    def temperature(self) -> float:
        """可学习温度参数（通过 log 参数化保证正值）。"""
        return self.log_temperature.exp().clamp(min=0.01, max=10.0).item()

    @property
    def temperature_tensor(self) -> torch.Tensor:
        """返回温度的 Tensor 形式（用于损失计算，保留梯度）。"""
        return self.log_temperature.exp().clamp(min=0.01, max=10.0)

    def _init_parameters(self):
        """用 Xavier 均匀初始化所有原型（后续会被 K-means++ 边缘点覆盖）。"""
        nn.init.xavier_uniform_(self.z_shared)
        for z in self.z_domain:
            nn.init.xavier_uniform_(z)

    def init_from_data(self, x_list: list, method: str = 'boundary_kmeans'):
        """用数据边缘点初始化原型（K-means++ 变体，选择距离均值最远的点）。

        Args:
            x_list : list of Tensor [N_d, D]，每个域的节点嵌入
            method : 'boundary_kmeans'（推荐）或 'random'
        """
        with torch.no_grad():
            # 统一在 CPU 上做初始化计算
            x_list_cpu = [x.cpu() for x in x_list]

            # 初始化共享原型：从所有域合并后选边缘点
            all_x = torch.cat(x_list_cpu, dim=0)  # [N_total, D]
            self.z_shared.data = self._boundary_kmeans_init(all_x, self.n_shared)

            # 初始化域特化原型：从各域单独选边缘点
            for d, x_d in enumerate(x_list_cpu):
                k_d = self._k_domain[d]
                if x_d.shape[0] >= k_d:
                    self.z_domain[d].data = self._boundary_kmeans_init(x_d, k_d)
                else:
                    idx = torch.randint(0, x_d.shape[0], (k_d,))
                    self.z_domain[d].data = x_d[idx].clone()

        print("[DA-AA] init done: K_s={}, K_d={}".format(self.n_shared, self._k_domain))

    def _boundary_kmeans_init(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """边缘点 K-means++ 初始化：优先选择距离数据均值最远的点。

        Args:
            x : [N, D]
            k : 原型数量

        Returns:
            centers : [k, D]
        """
        N, D = x.shape
        if N <= k:
            # 数据点不足，直接返回（可能有重复）
            idx = torch.arange(N, device=x.device)
            pad = torch.randint(0, N, (k - N,), device=x.device)
            return x[torch.cat([idx, pad])].clone()

        centers = []
        # 第一个中心：距离全局均值最远的点
        mean = x.mean(dim=0, keepdim=True)  # [1, D]
        dist_to_mean = torch.cdist(x, mean, p=2).squeeze(-1)  # [N]
        first_idx = dist_to_mean.argmax().item()
        centers.append(x[first_idx])

        # 后续中心：K-means++ 策略（距离已选中心最远）
        for _ in range(k - 1):
            c_stack = torch.stack(centers, dim=0)  # [c, D]
            dists = torch.cdist(x, c_stack, p=2).min(dim=-1).values  # [N]
            # 按距离平方采样（K-means++ 概率）
            probs = dists.pow(2)
            probs = probs / (probs.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1).item()
            centers.append(x[next_idx])

        return torch.stack(centers, dim=0)  # [k, D]

    def _get_domain_prototypes(self, domain_id: int) -> torch.Tensor:
        """获取域 d 的完整原型集合：[K_s + K_d, D]。"""
        z_dom = self.z_domain[domain_id]
        # 确保域特化原型与共享原型在同一设备（dynamic_expand 后可能不一致）
        if z_dom.device != self.z_shared.device:
            z_dom = z_dom.to(self.z_shared.device)
        return torch.cat([self.z_shared, z_dom], dim=0)

    def _ensure_device(self, device: torch.device):
        """将所有原型参数迁移到指定设备（用于懒迁移）。"""
        if self.z_shared.device != device:
            self.z_shared.data = self.z_shared.data.to(device)
        for i, z in enumerate(self.z_domain):
            if z.device != device:
                self.z_domain[i].data = z.data.to(device)

    def _compute_coefficients(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """计算重构系数 A（软单纯形投影）。

        使用归一化距离避免高维空间距离值过大导致 softmax 退化。

        Args:
            x : [N, D]
            z : [K, D]

        Returns:
            a : [N, K]  重构系数（非负，和为1）
        """
        # 确保 z 与 x 在同一设备
        z = z.to(x.device)

        # L2 归一化后计算余弦距离
        x_norm = F.normalize(x, dim=-1)
        z_norm = F.normalize(z, dim=-1)
        cos_sim = x_norm @ z_norm.t()        # [N, K]
        dist = 1.0 - cos_sim                 # [N, K]，值域 [0, 2]

        # 温度缩放 softmax
        tau = self.log_temperature.exp().clamp(min=0.01, max=10.0)
        a = F.softmax(-dist / tau, dim=-1)   # [N, K]，已满足非负且和为1

        if self.use_sparse_coeff:
            # Top-k 稀疏：保留最大的 k 个，其余置零后重归一化
            k = min(self.sparse_topk, a.shape[-1])
            topk_vals, topk_idx = torch.topk(a, k, dim=-1)
            a_sparse = torch.zeros_like(a)
            a_sparse.scatter_(-1, topk_idx, topk_vals)
            # 重归一化（softmax 后 top-k 截断需要重新归一化）
            a = a_sparse / (a_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        return a

    def forward(
        self,
        x: torch.Tensor,
        domain_id: int,
        x_struct: torch.Tensor = None,
        x_sem: torch.Tensor = None,
    ):
        """前向传播。

        Args:
            x         : [N, D]  节点嵌入
            domain_id : 当前域索引（0-indexed）
            x_struct  : [N, D_s]  结构嵌入（用于 HSIC，可选）
            x_sem     : [N, D_e]  语义嵌入（用于 HSIC，可选）

        Returns:
            x_out  : [N, D]  增强后的节点嵌入（残差形式）
            losses : dict    各损失分量
        """
        # 获取当前域的原型集合
        z = self._get_domain_prototypes(domain_id)  # [K_s+K_d, D]
        z = z.to(x.device)  # 确保设备一致

        # 计算重构系数
        a = self._compute_coefficients(x, z)  # [N, K_s+K_d]

        # 重构
        x_hat = a @ z  # [N, D]

        # 残差输出：固定混合系数 0.3，原型做轻量增量修正
        # 不用可学习系数，避免被 l_recon 拉到1导致完全替换原始嵌入
        x_out = x + 0.3 * (x_hat - x)

        # ---- 损失计算 ----
        losses = {}

        # 重构损失
        l_recon = F.mse_loss(x_hat, x.detach())
        losses['recon'] = l_recon

        # 顶点损失：原型间排斥力（欧氏距离倒数）
        # 距离越近惩罚越大，无论正交与否都有持续梯度
        # 与 l_recon（把原型拉向数据）平衡后，原型自然落在凸包顶点附近
        K_proto = z.shape[0]
        if K_proto > 1:
            dist_zz = torch.cdist(z, z, p=2)  # [K, K]
            mask_off = ~torch.eye(K_proto, dtype=torch.bool, device=z.device)
            l_vert = (1.0 / (dist_zz[mask_off] + 0.1)).mean()
        else:
            l_vert = torch.tensor(0.0, device=x.device)
        losses['vert'] = l_vert

        # 顶点偏离（仅监控）：原型到最近数据点的距离
        with torch.no_grad():
            dist_to_data = torch.cdist(z, x, p=2)
            losses['vert_dev_monitor'] = dist_to_data.min(dim=-1).values.mean()

        # 残差幅度惩罚：用 L1 惩罚重构误差，比 l_recon(L2) 对大误差更鲁棒
        l_res = (x_hat - x.detach()).abs().mean()
        losses['res'] = l_res

        # HSIC 解耦损失（可选）
        l_hsic = torch.tensor(0.0, device=x.device)
        if self._hsic_enabled and x_struct is not None and x_sem is not None:
            l_hsic = hsic_normalized(x_struct, x_sem)
            # 熔断机制：若 HSIC 值异常大，暂时禁用
            if l_hsic.item() > 0.9:
                self._hsic_fuse_count += 1
                if self._hsic_fuse_count >= self._hsic_fuse_window:
                    self._hsic_enabled = False
                    print("[DA-AA] HSIC fused: disabled after {} consecutive triggers".format(self._hsic_fuse_window))
                l_hsic = torch.tensor(0.0, device=x.device)
            else:
                self._hsic_fuse_count = 0
        losses['hsic'] = l_hsic

        # 梯度一致性损失（在 backward 后通过 compute_grad_consistency_loss 计算）
        losses['gc'] = torch.tensor(0.0, device=x.device)

        # 总损失：recon 主导，vert 排斥力防坍缩，res 控制残差幅度
        total_loss = (
            l_recon
            + self.lambda_vert * l_vert
            + self.lambda_res * l_res
            + self.lambda_hsic * l_hsic
        )
        losses['total'] = total_loss

        # 更新使用率 EMA（用于动态管理）
        with torch.no_grad():
            k_s = self.n_shared
            a_dom = a[:, k_s:]  # [N, K_d]
            usage = a_dom.mean(dim=0).cpu()  # [K_d]
            ema = self._usage_ema[domain_id]
            # 若 ema 大小与当前 K_d 不一致（如刚加载 checkpoint），自动重置
            if ema.shape[0] != usage.shape[0]:
                ema = torch.zeros_like(usage)
                self._usage_ema[domain_id] = ema
            self._usage_ema[domain_id] = self._ema_decay * ema + (1 - self._ema_decay) * usage

        return x_out, losses

    def compute_grad_consistency_loss(self) -> torch.Tensor:
        """计算梯度一致性损失（需在 backward 后调用）。

        通过 hook 收集各域对共享原型的梯度，计算域间余弦相似度。

        Returns:
            loss : scalar（越大表示各域梯度越不一致）
        """
        if len(self._shared_grads) < 2:
            return torch.tensor(0.0)

        grads = [g.flatten() for g in self._shared_grads if g is not None]
        if len(grads) < 2:
            self._shared_grads.clear()
            return torch.tensor(0.0)

        # 归一化梯度
        grads_norm = [F.normalize(g.unsqueeze(0), dim=-1) for g in grads]
        grads_stack = torch.cat(grads_norm, dim=0)  # [M, K_s*D]

        # 计算所有域对的余弦相似度均值
        sim_matrix = grads_stack @ grads_stack.t()  # [M, M]
        M = sim_matrix.shape[0]
        # 取上三角（不含对角线）
        mask = torch.triu(torch.ones(M, M, device=sim_matrix.device), diagonal=1).bool()
        cos_sims = sim_matrix[mask]

        if cos_sims.numel() == 0:
            self._shared_grads.clear()
            return torch.tensor(0.0)

        cos_mean = cos_sims.mean()
        # 损失：惩罚低一致性（cos_mean < 0.5 时产生正损失）
        loss = F.relu(0.5 - cos_mean)
        self._shared_grads.clear()
        return loss

    def register_grad_hooks(self):
        """注册梯度 hook，按域收集共享原型的梯度。
        
        每个 epoch 调用一次，清空上一轮收集的梯度。
        hook 会在本 epoch 的所有域 backward 后积累各域梯度。
        """
        self._shared_grads.clear()
        self._hook_handles = getattr(self, '_hook_handles', [])
        # 移除旧 hook，避免重复注册叠加
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        if self.z_shared.requires_grad:
            handle = self.z_shared.register_hook(
                lambda grad: self._shared_grads.append(grad.detach().clone())
            )
            self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # 动态原型管理
    # ------------------------------------------------------------------

    def dynamic_expand(self, domain_id: int, x: torch.Tensor, threshold: float = 0.3,
                       coverage_threshold: float = 0.15):
        """动态扩展域特化原型（每20 epoch 调用一次）。

        若某域中 max_a < threshold 的节点占比 > coverage_threshold，
        且当前原型数 < max_k_domain，则添加误差最大节点作为新原型。

        Args:
            domain_id : 域索引
            x         : [N, D]  当前域节点嵌入
            threshold : 最大系数阈值（低于此值认为"覆盖不足"）
            coverage_threshold : 覆盖不足节点占比阈值
        """
        if self._k_domain[domain_id] >= self.max_k_domain:
            return

        with torch.no_grad():
            z = self._get_domain_prototypes(domain_id)
            a = self._compute_coefficients(x, z)
            max_a = a.max(dim=-1).values  # [N]

            # 覆盖不足的节点比例
            low_coverage_ratio = (max_a < threshold).float().mean().item()

            if low_coverage_ratio > coverage_threshold:
                # 找误差最大的节点
                x_hat = a @ z
                errors = (x_hat - x).pow(2).sum(dim=-1)  # [N]
                worst_idx = errors.argmax().item()

                # 目标设备：与 z_shared 保持一致
                target_device = self.z_shared.device

                new_proto = x[worst_idx].detach().to(target_device)
                old_mean = self.z_domain[domain_id].data.to(target_device).mean(dim=0)
                new_proto_smooth = 0.9 * old_mean + 0.1 * new_proto

                old_data = self.z_domain[domain_id].data.to(target_device)
                new_data = torch.cat([old_data, new_proto_smooth.unsqueeze(0)], dim=0)
                self.z_domain[domain_id] = nn.Parameter(new_data)
                self._k_domain[domain_id] += 1

                # 扩展 EMA
                old_ema = self._usage_ema[domain_id]
                self._usage_ema[domain_id] = torch.cat([old_ema, torch.zeros(1)], dim=0)

                print("[DA-AA] domain {} expand: {} -> {}".format(
                    domain_id, self._k_domain[domain_id]-1, self._k_domain[domain_id]))

    def dynamic_prune(self, domain_id: int, min_usage: float = 0.01, min_k: int = 3):
        """动态修剪低使用率原型（每50 epoch 调用一次）。

        Args:
            domain_id : 域索引
            min_usage : 最低使用率阈值（低于此值的原型被删除）
            min_k     : 每域最少保留的原型数
        """
        if self._k_domain[domain_id] <= min_k:
            return

        with torch.no_grad():
            usage = self._usage_ema[domain_id]  # [K_d]
            keep_mask = usage >= min_usage

            # 至少保留 min_k 个
            n_keep = keep_mask.sum().item()
            if n_keep < min_k:
                # 保留使用率最高的 min_k 个
                _, top_idx = usage.topk(min_k)
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[top_idx] = True

            if keep_mask.all():
                return  # 无需修剪

            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            old_k = self._k_domain[domain_id]
            # 保持原设备
            new_data = self.z_domain[domain_id].data[keep_idx]
            self.z_domain[domain_id] = nn.Parameter(new_data)
            self._k_domain[domain_id] = new_data.shape[0]
            self._usage_ema[domain_id] = usage[keep_idx.cpu()]

            print("[DA-AA] domain {} prune: {} -> {}".format(domain_id, old_k, self._k_domain[domain_id]))

    def get_k_domain(self, domain_id: int) -> int:
        """返回域 d 当前的特化原型数量。"""
        return self._k_domain[domain_id]

    def get_usage_ema(self, domain_id: int) -> torch.Tensor:
        """返回域 d 的原型使用率 EMA。"""
        return self._usage_ema[domain_id]

    def extra_repr(self) -> str:
        return (
            f"n_domains={self.n_domains}, n_shared={self.n_shared}, "
            f"k_domain={self._k_domain}, dim={self.dim}, "
            f"temperature={self.temperature:.4f}"
        )
