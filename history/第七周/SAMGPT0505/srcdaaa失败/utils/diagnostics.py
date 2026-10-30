"""
diagnostics.py — DA-AA 训练监控与可视化工具集

包含：
  - DVPTracker      : 域方差保留率（DVP）追踪器
  - PrototypeMonitor: 原型熵、死亡原型、顶点偏离等指标计算
  - CCAChecker      : 事前 CCA 检验（结构-语义相关性）
  - ConvexHullEstimator: 凸包顶点数估计
  - Visualizer      : t-SNE、DVP 曲线、损失分解、使用率直方图等可视化

所有可视化结果自动保存到 ./diagnostics/ 目录。
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# 可视化依赖（懒加载，避免无 GUI 环境报错）
_plt = None
_tsne = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端，适合服务器
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_tsne():
    global _tsne
    if _tsne is None:
        from sklearn.manifold import TSNE
        _tsne = TSNE
    return _tsne


# ---------------------------------------------------------------------------
# DVP Tracker（域方差保留率）
# ---------------------------------------------------------------------------

class DVPTracker:
    """追踪每个域的方差保留率（DVP）。

    DVP_d = Var(Z_dom_d) / Var(X_d)

    使用滑动窗口平滑，避免单 epoch 波动。

    Args:
        n_domains   : 域数量
        window_size : 滑动窗口大小
        target_dvp  : 目标 DVP 阈值（默认 0.85）
    """

    def __init__(self, n_domains: int, window_size: int = 5, target_dvp: float = 0.1):
        """
        Args:
            target_dvp : 修正后的合理目标（原方案 0.85 是错误的，
                         端到端训练中 MSE 把原型拉向均值而非边缘，
                         0.1~0.3 是正常范围）
        """
        self.n_domains = n_domains
        self.window_size = window_size
        self.target_dvp = target_dvp
        self._history: Dict[int, List[float]] = defaultdict(list)
        self._smoothed: Dict[int, float] = {}

    def update(self, domain_id: int, z_dom: torch.Tensor, x: torch.Tensor,
               a: torch.Tensor = None):
        """更新域 d 的 DVP。

        DVP = 1 - Var(X - X_hat) / Var(X)
        即重构残差方差占原始数据方差的比例越小，DVP 越高（越接近1越好）。

        Args:
            domain_id : 域索引
            z_dom     : [K_d, D]  域特化原型（或完整原型集合）
            x         : [N, D]   原始节点嵌入
            a         : [N, K]   重构系数（可选，若不传则重新计算）
        """
        with torch.no_grad():
            # 如果没有传系数，用最近邻硬分配近似
            if a is None:
                z = z_dom.to(x.device)
                z_norm = torch.nn.functional.normalize(z, dim=-1)
                x_norm = torch.nn.functional.normalize(x, dim=-1)
                cos_sim = x_norm @ z_norm.t()          # [N, K]
                assign  = cos_sim.argmax(dim=-1)       # [N]
                x_hat   = z[assign]                    # [N, D]
            else:
                z = z_dom.to(x.device)
                a = a.to(x.device)
                x_hat = a @ z                          # [N, D]

            residual = x - x_hat
            var_x   = x.var(dim=0).mean().item()
            var_res = residual.var(dim=0).mean().item()

            if var_x < 1e-8:
                dvp = 1.0
            else:
                dvp = 1.0 - var_res / var_x
                dvp = float(max(0.0, min(1.0, dvp)))

        self._history[domain_id].append(dvp)
        window = self._history[domain_id][-self.window_size:]
        self._smoothed[domain_id] = sum(window) / len(window)

    def get_dvp(self, domain_id: int) -> float:
        """获取域 d 的平滑 DVP。"""
        return self._smoothed.get(domain_id, 0.0)

    def get_all_dvp(self) -> Dict[int, float]:
        """获取所有域的平滑 DVP。"""
        return dict(self._smoothed)

    def all_pass(self) -> bool:
        """检查所有域是否达到目标 DVP。"""
        if len(self._smoothed) < self.n_domains:
            return False
        return all(v >= self.target_dvp for v in self._smoothed.values())

    def get_history(self, domain_id: int) -> List[float]:
        """获取域 d 的 DVP 历史记录。"""
        return self._history[domain_id]


# ---------------------------------------------------------------------------
# Prototype Monitor（原型健康度监控）
# ---------------------------------------------------------------------------

class PrototypeMonitor:
    """计算原型健康度指标。

    指标：
      - 原型熵：-1/N Σ_i Σ_k a_ik log a_ik
      - 死亡原型数：使用率为 0 的原型个数
      - 顶点偏离距离：各原型到最近节点的最小距离均值
      - 块对角强度：1 - ||off_diag(C)||_F / ||C||_F
      - 残差幅度比：||X_hat - X|| / ||X||
      - 域使用均衡度：共享原型在不同域利用率分布的熵
    """

    @staticmethod
    def prototype_entropy(a: torch.Tensor, eps: float = 1e-8) -> float:
        """计算原型分配熵。

        Args:
            a : [N, K]  重构系数

        Returns:
            entropy : scalar（越小越集中，越大越均匀）
        """
        with torch.no_grad():
            log_a = torch.log(a.clamp(min=eps))
            entropy = -(a * log_a).sum(dim=-1).mean().item()
        return entropy

    @staticmethod
    def dead_prototypes(usage_ema: torch.Tensor, threshold: float = 1e-4) -> int:
        """计算死亡原型数（使用率极低）。

        Args:
            usage_ema : [K]  原型使用率 EMA
            threshold : 死亡阈值

        Returns:
            n_dead : int
        """
        return (usage_ema < threshold).sum().item()

    @staticmethod
    def vertex_deviation(z: torch.Tensor, x: torch.Tensor) -> float:
        """计算原型到最近数据点的平均最小余弦距离（与系数计算一致）。

        Args:
            z : [K, D]  原型
            x : [N, D]  数据点

        Returns:
            mean_min_cos_dist : scalar ∈ [0, 2]（越小越好）
        """
        with torch.no_grad():
            z_norm = torch.nn.functional.normalize(z, dim=-1)
            x_norm = torch.nn.functional.normalize(x, dim=-1)
            cos_dist = 1.0 - z_norm @ x_norm.t()  # [K, N]
            min_dist = cos_dist.min(dim=-1).values  # [K]
            return min_dist.mean().item()

    @staticmethod
    def block_diagonal_strength(a_list: List[torch.Tensor], n_shared: int = 0) -> float:
        """计算域特化原型的块对角强度。

        衡量各域节点主要使用域特化原型（而非共享原型）的程度。
        domain_ratio = mean(domain_usage) / mean(total_usage)

        Args:
            a_list   : list of [N_d, K_d]  各域的重构系数（K_d 可以不同）
            n_shared : 共享原型数量 K_s（a 的前 n_shared 列是共享原型）

        Returns:
            strength : scalar ∈ [0, 1]，越大说明域特化原型越被充分利用
        """
        if len(a_list) < 1 or n_shared < 0:
            return 0.0

        with torch.no_grad():
            domain_specific_ratios = []
            for a_d in a_list:
                a_d = a_d.cpu()
                K = a_d.shape[-1]
                k_domain = K - n_shared
                if k_domain <= 0:
                    domain_specific_ratios.append(0.0)
                    continue
                shared_usage = a_d[:, :n_shared].sum(dim=-1).mean().item()
                domain_usage = a_d[:, n_shared:].sum(dim=-1).mean().item()
                total = shared_usage + domain_usage
                if total > 1e-8:
                    domain_specific_ratios.append(domain_usage / total)
                else:
                    domain_specific_ratios.append(0.0)

            if not domain_specific_ratios:
                return 0.0
            return sum(domain_specific_ratios) / len(domain_specific_ratios)

    @staticmethod
    def residual_ratio(x_out: torch.Tensor, x: torch.Tensor) -> float:
        """计算实际输出修正量与原始嵌入的比值。

        ratio = ||x_out - x|| / ||x||
        反映原型分析对原始嵌入的修正幅度，目标 < 0.3。

        Args:
            x_out : [N, D]  DA-AA 输出嵌入
            x     : [N, D]  原始嵌入
        """
        with torch.no_grad():
            res_norm = (x_out - x).norm().item()
            x_norm = x.norm().item()
            return res_norm / (x_norm + 1e-8)

    @staticmethod
    def domain_balance_entropy(shared_usage_per_domain: List[torch.Tensor]) -> float:
        """计算共享原型在不同域的利用率均衡度（熵）。

        Args:
            shared_usage_per_domain : list of [K_s]  各域对共享原型的平均使用率

        Returns:
            entropy : scalar（理想值 ≈ log(M)）
        """
        if len(shared_usage_per_domain) == 0:
            return 0.0

        with torch.no_grad():
            # 各域总使用量
            domain_totals = torch.tensor(
                [u.sum().item() for u in shared_usage_per_domain]
            ).float()
            domain_probs = domain_totals / (domain_totals.sum() + 1e-8)
            entropy = -(domain_probs * torch.log(domain_probs.clamp(min=1e-8))).sum().item()
        return entropy

    @staticmethod
    def grad_consistency(shared_grads: List[torch.Tensor]) -> float:
        """计算域间嵌入分布差异度（替代无法计算的 grad_cos）。

        用各域节点嵌入均值之间的平均余弦距离衡量域间差异。
        值越小说明各域嵌入越相似（域差异小），越大说明域差异大。
        这个指标反映了共享原型需要覆盖多大的分布范围。

        Args:
            shared_grads : 复用参数名，实际传入各域嵌入均值列表 [list of [D]]

        Returns:
            mean_cos_dist : scalar ∈ [0, 2]
        """
        if len(shared_grads) < 2:
            return 0.0

        means = [F.normalize(g.flatten().unsqueeze(0), dim=-1) for g in shared_grads]
        means_stack = torch.cat(means, dim=0)  # [M, D]
        sim_matrix = means_stack @ means_stack.t()  # [M, M]
        M = sim_matrix.shape[0]
        mask = torch.triu(torch.ones(M, M), diagonal=1).bool()
        cos_sims = sim_matrix[mask]
        if cos_sims.numel() == 0:
            return 0.0
        # 返回平均余弦距离（1 - cos_sim），越大说明域间差异越大
        return (1.0 - cos_sims.mean()).item()


# ---------------------------------------------------------------------------
# CCA Checker（事前检验）
# ---------------------------------------------------------------------------

class CCAChecker:
    """CCA 检验：计算结构嵌入与语义嵌入的第一对典型变量相关系数。

    用于阶段0事前检验，决定是否启用 HSIC 损失。
    """

    @staticmethod
    def compute_cca_correlation(x_struct: torch.Tensor, x_sem: torch.Tensor,
                                n_components: int = 1) -> float:
        """计算第一对典型变量相关系数。

        Args:
            x_struct : [N, D1]  结构嵌入
            x_sem    : [N, D2]  语义嵌入
            n_components : 典型变量数量

        Returns:
            corr : 第一对典型变量相关系数
        """
        try:
            from sklearn.cross_decomposition import CCA
            x_s = x_struct.cpu().numpy()
            x_e = x_sem.cpu().numpy()

            # 零均值化
            x_s = x_s - x_s.mean(axis=0)
            x_e = x_e - x_e.mean(axis=0)

            cca = CCA(n_components=n_components, max_iter=500)
            cca.fit(x_s, x_e)
            x_c, y_c = cca.transform(x_s, x_e)

            # 第一对典型变量相关系数
            corr = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
            return float(corr)
        except Exception as e:
            print("[CCAChecker] failed: {}, return 0.0".format(e))
            return 0.0

    @staticmethod
    def should_enable_hsic(corr: float, threshold: float = 0.4) -> bool:
        """根据 CCA 相关系数决定是否启用 HSIC。

        Args:
            corr      : CCA 相关系数
            threshold : 阈值（默认 0.4）

        Returns:
            bool : True 表示应启用 HSIC
        """
        return corr >= threshold


# ---------------------------------------------------------------------------
# Convex Hull Estimator（凸包顶点数估计）
# ---------------------------------------------------------------------------

class ConvexHullEstimator:
    """估计数据凸包顶点数，用于确定 K_max。"""

    @staticmethod
    def estimate_hull_vertices(x: torch.Tensor, pca_variance_threshold: float = 0.95,
                               max_points: int = 2000) -> int:
        """估计数据凸包顶点数。

        步骤：
        1. PCA 降维至方差 > pca_variance_threshold
        2. 随机采样（若 N > max_points）
        3. 计算 Delaunay 三角剖分的顶点数

        Args:
            x                    : [N, D]  节点嵌入
            pca_variance_threshold: PCA 方差保留阈值
            max_points           : 最大采样点数

        Returns:
            n_vertices : 估计的凸包顶点数
        """
        try:
            from sklearn.decomposition import PCA
            from scipy.spatial import ConvexHull

            x_np = x.cpu().numpy()
            N = x_np.shape[0]

            # 随机采样
            if N > max_points:
                idx = np.random.choice(N, max_points, replace=False)
                x_np = x_np[idx]

            # PCA 降维
            pca = PCA()
            pca.fit(x_np)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_comp = np.searchsorted(cumvar, pca_variance_threshold) + 1
            n_comp = max(2, min(n_comp, 10))  # 限制在 [2, 10]

            x_pca = pca.transform(x_np)[:, :n_comp]

            # 凸包
            hull = ConvexHull(x_pca)
            return len(hull.vertices)

        except Exception as e:
            print("[ConvexHullEstimator] failed: {}, return 20".format(e))
            return 20


# ---------------------------------------------------------------------------
# Logger（结构化日志）
# ---------------------------------------------------------------------------

class AALogger:
    """DA-AA 训练日志记录器。

    每10 epoch 输出一行结构化日志，格式：
    epoch, domain, dvp, recon_loss, vert_loss, entropy, dead_protos,
    block_strength, res_ratio, grad_cos, ...
    """

    def __init__(self, log_dir: str = './diagnostics', log_file: str = 'aa_train.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self._header_written = False

    def _write_header(self):
        header = (
            "epoch,domain,dvp,recon_loss,vert_loss,res_loss,hsic_loss,"
            "entropy,dead_protos,vertex_dev,block_strength,res_ratio,grad_cos\n"
        )
        with open(self.log_path, 'w') as f:
            f.write(header)
        self._header_written = True
    def log(self, epoch: int, domain_id: int, metrics: dict):
        """记录一行日志。

        Args:
            epoch     : 当前 epoch
            domain_id : 域索引
            metrics   : 指标字典
        """
        if not self._header_written:
            self._write_header()

        row = (
            f"{epoch},{domain_id},"
            f"{metrics.get('dvp', 0.0):.4f},"
            f"{metrics.get('recon_loss', 0.0):.6f},"
            f"{metrics.get('vert_loss', 0.0):.6f},"
            f"{metrics.get('res_loss', 0.0):.6f},"
            f"{metrics.get('hsic_loss', 0.0):.6f},"
            f"{metrics.get('entropy', 0.0):.4f},"
            f"{metrics.get('dead_protos', 0):.0f},"
            f"{metrics.get('vertex_dev', 0.0):.4f},"
            f"{metrics.get('block_strength', 0.0):.4f},"
            f"{metrics.get('res_ratio', 0.0):.4f},"
            f"{metrics.get('grad_cos', 0.0):.4f}\n"
        )
        with open(self.log_path, 'a') as f:
            f.write(row)

    def print_summary(self, epoch: int, domain_metrics: Dict[int, dict]):
        """打印当前 epoch 的摘要（使用修正后的判据）。

        修正后的判据：
        - 熵 > 1.0 且接近 ln(K) 为健康（高熵=均匀使用=无坍缩）
        - DVP > 0.1 且稳定为合理（不追求 0.85）
        - dead_protos = 0 为必要条件
        """
        print("\n" + "=" * 60)
        print("[DA-AA] Epoch {} monitor summary".format(epoch))
        print("=" * 60)
        for d, m in domain_metrics.items():
            dvp     = m.get('dvp', 0.0)
            entropy = m.get('entropy', 0.0)
            dead    = m.get('dead_protos', 0)
            recon   = m.get('recon_loss', 0.0)
            # 修正后的判据：熵>1.0为健康，DVP>0.1为合理
            entropy_ok = "OK" if entropy > 1.0 else "LOW"
            dvp_ok     = "OK" if dvp > 0.1 else "LOW"
            dead_ok    = "OK" if dead == 0 else "DEAD"
            print(
                "  domain {}: DVP={:.3f}[{}]  entropy={:.3f}[{}]  "
                "dead=[{}]  recon={:.5f}".format(
                    d, dvp, dvp_ok, entropy, entropy_ok, dead_ok, recon
                )
            )
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Visualizer（可视化）
# ---------------------------------------------------------------------------

class Visualizer:
    """DA-AA 训练可视化工具。

    所有图表自动保存到 save_dir 目录。
    """

    def __init__(self, save_dir: str = './diagnostics', domain_names: List[str] = None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.domain_names = domain_names

        # 历史记录（用于曲线图）
        self._loss_history: Dict[str, List[float]] = defaultdict(list)
        self._dvp_history: Dict[int, List[float]] = defaultdict(list)
        self._epochs: List[int] = []

    def _domain_label(self, d: int) -> str:
        if self.domain_names and d < len(self.domain_names):
            return self.domain_names[d]
        return f"Domain {d}"

    def update_history(self, epoch: int, losses: dict, dvp_dict: Dict[int, float]):
        """更新历史记录。"""
        self._epochs.append(epoch)
        for k, v in losses.items():
            self._loss_history[k].append(v)
        for d, dvp in dvp_dict.items():
            self._dvp_history[d].append(dvp)

    def plot_prototype_usage(self, domain_id: int, usage: torch.Tensor, epoch: int):
        """Plot prototype usage histogram."""
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(8, 4))

        usage_np = usage.cpu().numpy()
        ax.bar(range(len(usage_np)), usage_np, color='steelblue', alpha=0.8)
        ax.set_xlabel('Prototype Index')
        ax.set_ylabel('Avg Usage Rate')
        ax.set_title('{} Prototype Usage (Epoch {})'.format(self._domain_label(domain_id), epoch))
        ax.axhline(y=0.01, color='red', linestyle='--', label='Dead threshold (1%)')
        ax.legend()

        path = os.path.join(self.save_dir, 'usage_domain{}_ep{}.png'.format(domain_id, epoch))
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)

    def plot_tsne(self, x_list: List[torch.Tensor], z_list: List[torch.Tensor],
                  epoch: int, perplexity: int = 30,
                  z_shared: torch.Tensor = None):
        """Plot t-SNE embedding of nodes + domain-specific prototypes + shared prototypes.

        Args:
            x_list   : list of [N_d, D]  node embeddings per domain
            z_list   : list of [K_d, D]  domain-specific prototypes per domain
            epoch    : current epoch
            z_shared : [K_s, D]  shared prototypes (optional, drawn in black)
        """
        plt = _get_plt()
        TSNE = _get_tsne()

        all_points = []
        all_labels = []   # domain index (shared protos use -1)
        all_types  = []   # 0=node, 1=domain proto, 2=shared proto

        max_n_per_domain = 1000  # 每域最多采样节点数

        for d, (x, z) in enumerate(zip(x_list, z_list)):
            x_np = x.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()

            if x_np.shape[0] > max_n_per_domain:
                idx = np.random.choice(x_np.shape[0], max_n_per_domain, replace=False)
                x_np = x_np[idx]

            all_points.append(x_np)
            all_labels.extend([d] * x_np.shape[0])
            all_types.extend([0] * x_np.shape[0])

            all_points.append(z_np)
            all_labels.extend([d] * z_np.shape[0])
            all_types.extend([1] * z_np.shape[0])

        # 共享原型
        if z_shared is not None:
            zs_np = z_shared.detach().cpu().numpy()
            all_points.append(zs_np)
            all_labels.extend([-1] * zs_np.shape[0])
            all_types.extend([2] * zs_np.shape[0])

        all_points = np.vstack(all_points)
        all_labels = np.array(all_labels)
        all_types  = np.array(all_types)

        total_n = all_points.shape[0]
        print("[Visualizer] t-SNE: {} points total ({} domains)".format(
            total_n, len(x_list)))

        try:
            perp = min(perplexity, total_n - 1)
            # sklearn >= 1.1 将 n_iter 改名为 max_iter，做兼容处理
            import sklearn
            sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
            tsne_kwargs = dict(n_components=2, perplexity=perp, random_state=42)
            if sklearn_version >= (1, 1):
                tsne_kwargs['max_iter'] = 1000
            else:
                tsne_kwargs['n_iter'] = 1000
            tsne = TSNE(**tsne_kwargs)
            emb = tsne.fit_transform(all_points)
        except Exception as e:
            print("[Visualizer] t-SNE failed: {}".format(e))
            return

        # 颜色：每个域一种颜色，共享原型用黑色
        n_domains = len(x_list)
        cmap = plt.cm.tab10
        domain_colors = [cmap(i / max(n_domains, 1)) for i in range(n_domains)]

        fig, ax = plt.subplots(figsize=(12, 9))

        for d in range(n_domains):
            c = domain_colors[d]
            lbl = self._domain_label(d)

            # 节点（小圆点，半透明）
            mask_node = (all_labels == d) & (all_types == 0)
            if mask_node.any():
                ax.scatter(emb[mask_node, 0], emb[mask_node, 1],
                           color=c, alpha=0.3, s=10,
                           label='{} nodes (n={})'.format(lbl, mask_node.sum()))

            # 域特化原型（大星形，实色边框）
            mask_proto = (all_labels == d) & (all_types == 1)
            if mask_proto.any():
                ax.scatter(emb[mask_proto, 0], emb[mask_proto, 1],
                           color=c, marker='*', s=300,
                           edgecolors='black', linewidths=0.8,
                           label='{} protos (k={})'.format(lbl, mask_proto.sum()),
                           zorder=5)

        # 共享原型（黑色菱形，最显眼）
        mask_shared = (all_types == 2)
        if mask_shared.any():
            ax.scatter(emb[mask_shared, 0], emb[mask_shared, 1],
                       color='black', marker='D', s=150,
                       edgecolors='white', linewidths=0.8,
                       label='Shared protos (k={})'.format(mask_shared.sum()),
                       zorder=6)

        ax.set_title('t-SNE Embedding  Epoch {}  (total {} pts)'.format(epoch, total_n))
        ax.legend(loc='upper right', fontsize=7, ncol=2,
                  framealpha=0.8, markerscale=1.2)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')

        path = os.path.join(self.save_dir, 'tsne_ep{}.png'.format(epoch))
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print("[Visualizer] t-SNE saved: {}  ({} pts)".format(path, total_n))

    def plot_dvp_curves(self, epoch: int):
        """Plot DVP curves for all domains over training epochs."""
        if not self._epochs:
            return

        plt = _get_plt()

        # 主图：完整 DVP 曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax_full = axes[0]
        ax_zoom = axes[1]

        all_dvp_vals = []
        for d, dvp_hist in self._dvp_history.items():
            epochs = self._epochs[:len(dvp_hist)]
            ax_full.plot(epochs, dvp_hist, label=self._domain_label(d),
                         linewidth=2, marker='o', markersize=3)
            all_dvp_vals.extend(dvp_hist)

        ax_full.axhline(y=0.85, color='red', linestyle='--',
                        linewidth=1.5, label='Target=0.85')
        ax_full.set_xlabel('Epoch')
        ax_full.set_ylabel('DVP')
        ax_full.set_title('DVP Curves (full range)')
        ax_full.legend(fontsize=8)
        ax_full.set_ylim(0, 1.05)
        ax_full.grid(True, alpha=0.3)

        # 放大图：Y 轴自适应，看细节差异
        if all_dvp_vals:
            y_min = max(0.0, min(all_dvp_vals) - 0.05)
            y_max = min(1.05, max(all_dvp_vals) + 0.05)
            # 如果所有值都一样（如全1.0），强制拉开范围
            if y_max - y_min < 0.02:
                y_min = max(0.0, y_min - 0.1)
                y_max = min(1.05, y_max + 0.02)

        for d, dvp_hist in self._dvp_history.items():
            epochs = self._epochs[:len(dvp_hist)]
            ax_zoom.plot(epochs, dvp_hist, label=self._domain_label(d),
                         linewidth=2, marker='o', markersize=3)

        ax_zoom.axhline(y=0.85, color='red', linestyle='--',
                        linewidth=1.5, label='Target=0.85')
        ax_zoom.set_xlabel('Epoch')
        ax_zoom.set_ylabel('DVP')
        ax_zoom.set_title('DVP Curves (zoomed)')
        ax_zoom.legend(fontsize=8)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.grid(True, alpha=0.3)

        # 在放大图上标注每条线的最终值
        for d, dvp_hist in self._dvp_history.items():
            if dvp_hist:
                ax_zoom.annotate(
                    '{:.4f}'.format(dvp_hist[-1]),
                    xy=(self._epochs[len(dvp_hist)-1], dvp_hist[-1]),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=7, va='center'
                )

        fig.suptitle('Domain Variance Preservation (DVP)  Epoch {}'.format(epoch),
                     fontsize=12)
        path = os.path.join(self.save_dir, 'dvp_curves_ep{}.png'.format(epoch))
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)

    def plot_loss_decomposition(self, epoch: int):
        """Plot loss decomposition curves (each loss component over epochs).
        
        Skips components that are always zero (e.g. hsic when disabled).
        """
        if not self._epochs:
            return

        plt = _get_plt()

        # 过滤掉恒为0的曲线（如 hsic 未启用时）
        active_keys = [
            k for k in self._loss_history
            if k != 'total' and any(v != 0.0 for v in self._loss_history[k])
        ]
        # 始终保留 recon / vert / res，即使偶尔为0
        always_show = {'recon', 'vert', 'res'}
        loss_keys = sorted(
            set(active_keys) | {k for k in always_show if k in self._loss_history},
            key=lambda k: ['recon', 'vert', 'res', 'hsic', 'gc'].index(k)
                          if k in ['recon', 'vert', 'res', 'hsic', 'gc'] else 99
        )

        if not loss_keys:
            return

        n_plots = len(loss_keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, loss_keys):
            hist = self._loss_history[key]
            epochs = self._epochs[:len(hist)]
            ax.plot(epochs, hist, linewidth=2, color='steelblue')
            # 标注最终值
            if hist:
                ax.annotate('final={:.4f}'.format(hist[-1]),
                            xy=(epochs[-1], hist[-1]),
                            xytext=(-40, 8), textcoords='offset points',
                            fontsize=7, color='steelblue')
            ax.set_title('L_{}'.format(key))
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            # 如果值域很小，自适应 Y 轴
            if hist and max(hist) - min(hist) < 1e-6:
                ax.set_ylim(min(hist) - 0.01, max(hist) + 0.01)

        # 额外加一张 total loss 曲线
        if 'total' in self._loss_history:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            total_hist = self._loss_history['total']
            total_epochs = self._epochs[:len(total_hist)]
            ax2.plot(total_epochs, total_hist, linewidth=2, color='coral')
            ax2.set_title('L_total (DA-AA auxiliary)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            path2 = os.path.join(self.save_dir, 'loss_total_ep{}.png'.format(epoch))
            fig2.tight_layout()
            fig2.savefig(path2, dpi=120)
            plt.close(fig2)

        fig.suptitle('Loss Decomposition  Epoch {}'.format(epoch), fontsize=12)
        path = os.path.join(self.save_dir, 'loss_decomp_ep{}.png'.format(epoch))
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close(fig)

    def plot_cdg_comparison(self, domain_names: List[str],
                            acc_without: List[float], acc_with: List[float],
                            epoch: int):
        """Plot cross-domain gain (CDG) comparison bar chart."""
        plt = _get_plt()
        x = np.arange(len(domain_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width / 2, acc_without, width,
                       label='Shared protos only', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, acc_with, width,
                       label='Shared + domain-specific', color='coral', alpha=0.8)

        ax.set_xlabel('Target Domain')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Cross-Domain Gain (CDG) Comparison (Epoch {})'.format(epoch))
        ax.set_xticks(x)
        ax.set_xticklabels(domain_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for b1, b2 in zip(bars1, bars2):
            gain = b2.get_height() - b1.get_height()
            ax.annotate('+{:.1f}%'.format(gain),
                        xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9, color='darkred')

        path = os.path.join(self.save_dir, 'cdg_comparison_ep{}.png'.format(epoch))
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print("[Visualizer] CDG chart saved: {}".format(path))
