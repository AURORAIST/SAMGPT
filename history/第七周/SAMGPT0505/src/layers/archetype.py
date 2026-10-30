"""
archetype.py — 域感知原型分析模块（Domain-Aware Archetypal Analysis, DA-AA）

自适应版本：根据实时 DVP 自动调整 λ_vert 和 λ_res，无需手工调度。

新增自适应参数：
    adaptive_vert   : 是否启用自动调节 λ_vert  (默认 True)
    target_dvp_low  : 低于此 DVP 时增加推力    (默认 0.10)
    target_dvp_high : 高于此 DVP 时减小推力并增加收缩 (默认 0.35)
    vert_lr         : λ_vert 调整步长          (默认 0.01)
    res_lr          : λ_res 调整步长           (默认 0.002)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 单纯形投影
# ---------------------------------------------------------------------------
def project_simplex(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    u, _ = torch.sort(v, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1)
    rho_candidates = u - (cssv - 1.0) / (torch.arange(1, v.shape[-1] + 1,
                                                        device=v.device, dtype=v.dtype))
    mask = rho_candidates > 0
    rho_idx = mask.long().flip(-1).argmax(dim=-1, keepdim=True)
    rho_idx = v.shape[-1] - 1 - rho_idx
    theta = (cssv.gather(-1, rho_idx) - 1.0) / (rho_idx.float() + 1.0)
    w = (v - theta).clamp(min=0.0)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    return w


# ---------------------------------------------------------------------------
# HSIC
# ---------------------------------------------------------------------------
def hsic_normalized(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    N = X.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=X.device)

    def rbf_kernel(A):
        dist = torch.cdist(A, A, p=2).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))

    def center_kernel(K):
        H = torch.eye(N, device=K.device) - 1.0 / N
        return H @ K @ H

    Kx = center_kernel(rbf_kernel(X))
    Ky = center_kernel(rbf_kernel(Y))
    hsic_val = (Kx * Ky).sum() / ((N - 1) ** 2 + 1e-8)
    norm_x = ((Kx * Kx).sum() / ((N - 1) ** 2 + 1e-8)).sqrt().clamp(min=1e-8)
    norm_y = ((Ky * Ky).sum() / ((N - 1) ** 2 + 1e-8)).sqrt().clamp(min=1e-8)
    return hsic_val / (norm_x * norm_y)


# ---------------------------------------------------------------------------
# 主模块
# ---------------------------------------------------------------------------
class DomainAwareAA(nn.Module):
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
        # ---------- 自适应参数 ----------
        adaptive_vert: bool = True,
        target_dvp_low: float = 0.10,
        target_dvp_high: float = 0.35,
        vert_lr: float = 0.01,
        res_lr: float = 0.002,
    ):
        super().__init__()

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

        init_temp = temperature if temperature is not None else 1.0
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp)))

        self.z_shared = nn.Parameter(torch.empty(n_shared, dim))
        self._k_domain = [n_domain] * n_domains
        self.z_domain = nn.ParameterList([
            nn.Parameter(torch.empty(n_domain, dim))
            for _ in range(n_domains)
        ])

        # HSIC 熔断
        self._hsic_enabled = (lambda_hsic > 0.0)
        self._hsic_fuse_count = 0
        self._hsic_fuse_window = 5

        # 动态管理统计
        self._usage_ema = [torch.zeros(n_domain) for _ in range(n_domains)]
        self._ema_decay = 0.99

        self._shared_grads = []

        # ---------- 自适应控制 ----------
        self.adaptive_vert = adaptive_vert
        self.target_dvp_low = target_dvp_low
        self.target_dvp_high = target_dvp_high
        self.vert_lr = vert_lr
        self.res_lr = res_lr
        # 用 buffer 保存移动平均的 DVP，以便在 state_dict 中保存
        self.register_buffer('_ema_dvp', torch.tensor(0.2))

        self._init_parameters()

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().clamp(min=0.01, max=10.0).item()

    @property
    def temperature_tensor(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=10.0)

    def _init_parameters(self):
        # 使用 Xavier 初始化，使原型从坐标原点附近开始（数据中心附近）
        nn.init.xavier_uniform_(self.z_shared)
        for z in self.z_domain:
            nn.init.xavier_uniform_(z)

    def init_from_data(self, x_list: list, method: str = 'boundary_kmeans'):
        with torch.no_grad():
            x_list_cpu = [x.cpu() for x in x_list]
            if method == 'gaussian':
                nn.init.normal_(self.z_shared, mean=0, std=0.1)
                for z in self.z_domain:
                    nn.init.normal_(z, mean=0, std=0.1)
                print("[DA-AA] init from Gaussian (mean=0, std=0.1)")
                return

            all_x = torch.cat(x_list_cpu, dim=0)
            self.z_shared.data = self._boundary_kmeans_init(all_x, self.n_shared)
            for d, x_d in enumerate(x_list_cpu):
                k_d = self._k_domain[d]
                if x_d.shape[0] >= k_d:
                    self.z_domain[d].data = self._boundary_kmeans_init(x_d, k_d)
                else:
                    idx = torch.randint(0, x_d.shape[0], (k_d,))
                    self.z_domain[d].data = x_d[idx].clone()
        print("[DA-AA] init done: K_s={}, K_d={}".format(self.n_shared, self._k_domain))

    def _boundary_kmeans_init(self, x: torch.Tensor, k: int) -> torch.Tensor:
        N, D = x.shape
        if N <= k:
            idx = torch.arange(N, device=x.device)
            pad = torch.randint(0, N, (k - N,), device=x.device)
            return x[torch.cat([idx, pad])].clone()
        centers = []
        mean = x.mean(dim=0, keepdim=True)
        dist_to_mean = torch.cdist(x, mean, p=2).squeeze(-1)
        first_idx = dist_to_mean.argmax().item()
        centers.append(x[first_idx])
        for _ in range(k - 1):
            c_stack = torch.stack(centers, dim=0)
            dists = torch.cdist(x, c_stack, p=2).min(dim=-1).values
            probs = dists.pow(2)
            probs = probs / (probs.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1).item()
            centers.append(x[next_idx])
        return torch.stack(centers, dim=0)

    def _get_domain_prototypes(self, domain_id: int) -> torch.Tensor:
        z_dom = self.z_domain[domain_id]
        if z_dom.device != self.z_shared.device:
            z_dom = z_dom.to(self.z_shared.device)
        return torch.cat([self.z_shared, z_dom], dim=0)

    def _ensure_device(self, device: torch.device):
        if self.z_shared.device != device:
            self.z_shared.data = self.z_shared.data.to(device)
        for i, z in enumerate(self.z_domain):
            if z.device != device:
                self.z_domain[i].data = z.data.to(device)

    def _compute_coefficients(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.to(x.device)
        x_norm = F.normalize(x, dim=-1)
        z_norm = F.normalize(z, dim=-1)
        cos_sim = x_norm @ z_norm.t()
        dist = 1.0 - cos_sim
        tau = self.log_temperature.exp().clamp(min=0.01, max=10.0)
        a = F.softmax(-dist / tau, dim=-1)
        if self.use_sparse_coeff:
            k = min(self.sparse_topk, a.shape[-1])
            topk_vals, topk_idx = torch.topk(a, k, dim=-1)
            a_sparse = torch.zeros_like(a)
            a_sparse.scatter_(-1, topk_idx, topk_vals)
            a = a_sparse / (a_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return a

    def forward(self, x, domain_id, x_struct=None, x_sem=None):
        z = self._get_domain_prototypes(domain_id)
        z = z.to(x.device)
        a = self._compute_coefficients(x, z)
        x_hat = a @ z
        x_out = x + 0.3 * (x_hat - x)

        losses = {}
        l_recon = F.mse_loss(x_hat, x.detach())
        losses['recon'] = l_recon

        K_proto = z.shape[0]
        if K_proto > 1:
            dist_zz = torch.cdist(z, z, p=2)
            mask_off = ~torch.eye(K_proto, dtype=torch.bool, device=z.device)
            l_vert = (1.0 / (dist_zz[mask_off] + 0.1)).mean()
        else:
            l_vert = torch.tensor(0.0, device=x.device)
        losses['vert'] = l_vert

        with torch.no_grad():
            dist_to_data = torch.cdist(z, x, p=2)
            losses['vert_dev_monitor'] = dist_to_data.min(dim=-1).values.mean()

        l_res = (x_hat - x.detach()).abs().mean()
        losses['res'] = l_res

        l_hsic = torch.tensor(0.0, device=x.device)
        if self._hsic_enabled and x_struct is not None and x_sem is not None:
            l_hsic = hsic_normalized(x_struct, x_sem)
            if l_hsic.item() > 0.9:
                self._hsic_fuse_count += 1
                if self._hsic_fuse_count >= self._hsic_fuse_window:
                    self._hsic_enabled = False
                    print("[DA-AA] HSIC fused")
                l_hsic = torch.tensor(0.0, device=x.device)
            else:
                self._hsic_fuse_count = 0
        losses['hsic'] = l_hsic

        losses['gc'] = torch.tensor(0.0, device=x.device)

        total_loss = (
            l_recon
            + self.lambda_vert * l_vert
            + self.lambda_res * l_res
            + self.lambda_hsic * l_hsic
        )
        losses['total'] = total_loss

        # ----- 更新使用率 EMA -----
        with torch.no_grad():
            k_s = self.n_shared
            a_dom = a[:, k_s:]
            usage = a_dom.mean(dim=0).cpu()
            ema = self._usage_ema[domain_id]
            if ema.shape[0] != usage.shape[0]:
                ema = torch.zeros_like(usage)
                self._usage_ema[domain_id] = ema
            self._usage_ema[domain_id] = self._ema_decay * ema + (1 - self._ema_decay) * usage

        # ----- 自适应调节 λ_vert 和 λ_res -----
        if self.adaptive_vert and self.training:
            with torch.no_grad():
                # 计算共享原型的瞬时 DVP
                z_s = self.z_shared
                if z_s.shape[0] > 1:
                    dvp_now = z_s.var(dim=0).mean()
                else:
                    dvp_now = torch.tensor(0.0, device=x.device)
                # 指数移动平均
                self._ema_dvp = 0.9 * self._ema_dvp + 0.1 * dvp_now

                # 根据 EMA DVP 调整 lambda_vert
                if self._ema_dvp < self.target_dvp_low:
                    self.lambda_vert += self.vert_lr
                elif self._ema_dvp > self.target_dvp_high:
                    self.lambda_vert -= self.vert_lr
                self.lambda_vert = float(
                    max(0.001, min(0.5, self.lambda_vert))
                )

                # 根据 EMA DVP 调整 lambda_res
                if self._ema_dvp > self.target_dvp_high * 1.2:
                    self.lambda_res += self.res_lr
                elif self._ema_dvp < self.target_dvp_low * 1.2:
                    self.lambda_res -= self.res_lr
                self.lambda_res = float(
                    max(0.005, min(0.5, self.lambda_res))
                )

        return x_out, losses

    def compute_grad_consistency_loss(self) -> torch.Tensor:
        if len(self._shared_grads) < 2:
            return torch.tensor(0.0)
        grads = [g.flatten() for g in self._shared_grads if g is not None]
        if len(grads) < 2:
            self._shared_grads.clear()
            return torch.tensor(0.0)
        grads_norm = [F.normalize(g.unsqueeze(0), dim=-1) for g in grads]
        grads_stack = torch.cat(grads_norm, dim=0)
        sim_matrix = grads_stack @ grads_stack.t()
        M = sim_matrix.shape[0]
        mask = torch.triu(torch.ones(M, M, device=sim_matrix.device), diagonal=1).bool()
        cos_sims = sim_matrix[mask]
        if cos_sims.numel() == 0:
            self._shared_grads.clear()
            return torch.tensor(0.0)
        cos_mean = cos_sims.mean()
        loss = F.relu(0.5 - cos_mean)
        self._shared_grads.clear()
        return loss

    def register_grad_hooks(self):
        self._shared_grads.clear()
        self._hook_handles = getattr(self, '_hook_handles', [])
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if self.z_shared.requires_grad:
            handle = self.z_shared.register_hook(
                lambda grad: self._shared_grads.append(grad.detach().clone())
            )
            self._hook_handles.append(handle)

    # ---------- 动态管理 ----------
    def dynamic_expand(self, domain_id: int, x: torch.Tensor, threshold: float = 0.3,
                       coverage_threshold: float = 0.15):
        if self._k_domain[domain_id] >= self.max_k_domain:
            return
        with torch.no_grad():
            z = self._get_domain_prototypes(domain_id)
            a = self._compute_coefficients(x, z)
            max_a = a.max(dim=-1).values
            low_coverage_ratio = (max_a < threshold).float().mean().item()
            if low_coverage_ratio > coverage_threshold:
                x_hat = a @ z
                errors = (x_hat - x).pow(2).sum(dim=-1)
                worst_idx = errors.argmax().item()
                target_device = self.z_shared.device
                new_proto = x[worst_idx].detach().to(target_device)
                old_mean = self.z_domain[domain_id].data.to(target_device).mean(dim=0)
                new_proto_smooth = 0.9 * old_mean + 0.1 * new_proto
                old_data = self.z_domain[domain_id].data.to(target_device)
                new_data = torch.cat([old_data, new_proto_smooth.unsqueeze(0)], dim=0)
                self.z_domain[domain_id] = nn.Parameter(new_data)
                self._k_domain[domain_id] += 1
                old_ema = self._usage_ema[domain_id]
                self._usage_ema[domain_id] = torch.cat([old_ema, torch.zeros(1)], dim=0)
                print("[DA-AA] domain {} expand: {} -> {}".format(
                    domain_id, self._k_domain[domain_id]-1, self._k_domain[domain_id]))

    def dynamic_prune(self, domain_id: int, min_usage: float = 0.01, min_k: int = 3):
        if self._k_domain[domain_id] <= min_k:
            return
        with torch.no_grad():
            usage = self._usage_ema[domain_id]
            keep_mask = usage >= min_usage
            n_keep = keep_mask.sum().item()
            if n_keep < min_k:
                _, top_idx = usage.topk(min_k)
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[top_idx] = True
            if keep_mask.all():
                return
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            old_k = self._k_domain[domain_id]
            new_data = self.z_domain[domain_id].data[keep_idx]
            self.z_domain[domain_id] = nn.Parameter(new_data)
            self._k_domain[domain_id] = new_data.shape[0]
            self._usage_ema[domain_id] = usage[keep_idx.cpu()]
            print("[DA-AA] domain {} prune: {} -> {}".format(domain_id, old_k, self._k_domain[domain_id]))

    def get_k_domain(self, domain_id: int) -> int:
        return self._k_domain[domain_id]

    def get_usage_ema(self, domain_id: int) -> torch.Tensor:
        return self._usage_ema[domain_id]

    def extra_repr(self) -> str:
        return (
            f"n_domains={self.n_domains}, n_shared={self.n_shared}, "
            f"k_domain={self._k_domain}, dim={self.dim}, "
            f"temperature={self.temperature:.4f}, "
            f"adaptive_vert={self.adaptive_vert}"
        )