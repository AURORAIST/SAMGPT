import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, GcnLayers, MLP, GatLayers
from layers import AvgReadout 
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from layers.prompt import *
import copy

# ------------------------------------------------------------
# preprompt.py：预训练阶段的核心模型 PrePrompt。
#
# 该模块的要点：
# 1) 预训练任务支持两类：
#    - GRAPHCL：需要两份增强视图 + DGI/GraphCL 类对比损失（BCEWithLogitsLoss）
#
# 2) prompt 分为两种：
#    - feature_prompt_layers：对输入特征 seq 做 add/mul 调制
#    - structure_prompt_layers：作为“层 prompt”传给 GNN 的每一层（num_layers_num 层）
#
# 3) 多数据集预训练：对每个 pretrain dataset 都有一套 prompt 参数（ModuleList 按数据集索引）。
#
# 4) ablation：控制仅使用 fea prompting / 仅使用 str prompting / 两者融合等。
# ------------------------------------------------------------


class PrePrompt(nn.Module):
    """预训练 Prompt 模型。

    Args:
        n_in: 输入特征维度（PCA 后/统一维度 unify_dim）
        n_h: 隐层维度（GNN 输出维度）
        activation: 非线性（GraphCL/DGI 内部可能用到）
        num_pretrain_dataset_num: 预训练数据集数量（或参与预训练的图数量）
        num_layers_num: GNN 层数
        dropout: GNN dropout
        type_: prompt 的组合类型（'add'/'mul'），传给 textprompt
        backbone: 'gcn' 或 'gat'
        alpha: fea/str 融合系数（combine）
        ablation: 消融选项（'all','ft','st','None',...）
    """

    def __init__(
        self,
        n_in,
        n_h,
        activation,
        num_pretrain_dataset_num,
        num_layers_num,
        dropout,
        type_,
        backbone='gcn',
        alpha=1.0,
        ablation='all',
        enable_cluster_enhance=False,
        intra_clusters=8,
        shared_prototypes_num=16,
        cluster_interval=1,
        cluster_tau=0.2,
        lambda_cross=0.1,
        lambda_reg=0.01,
        lambda_proto=0.05,
        cluster_conf_threshold=0.6,
        prototype_ema_momentum=0.95,
    ):
        super(PrePrompt, self).__init__()

        # 预训练目标封装
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)

        # 图级 readout（平均池化）
        self.read = AvgReadout()

        self.prompttype = type_

        # ------------------- prompt 参数（按数据集分组）-------------------
        # 每个数据集一套 feature prompt（形状 [1, n_in]）
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)])

        # 每个数据集一套 structure prompt：每层一个 textprompt（通常为 [1, n_h]）
        self.structure_prompt_layers = nn.ModuleList(
            [nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)]) for _ in range(num_pretrain_dataset_num)]
        )

        # ------------------- 主干 GNN -------------------
        # 默认使用 GCN；可选 GAT
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if backbone == 'gat':
            self.gcn = GatLayers(n_in, n_h, num_layers_num, dropout)

            # GAT 多头时，每层输出维度通常变成 n_h * heads（concat=True 的情形）
            str_prompt = [textprompt(n_h * self.gcn.heads, type_) for _ in range(num_layers_num)]
            # str_prompt.append(textprompt(n_h, type_))

            # 重新构造 structure_prompt_layers：每个数据集拷贝一份同构 prompt 列表
            self.structure_prompt_layers = nn.ModuleList([nn.ModuleList(copy.deepcopy(str_prompt)) for _ in range(num_pretrain_dataset_num)])

        # combine: 融合 fea/str 的系数（在 ablation('all') 下生效）
        self.combine = alpha

        # GRAPHCL/DGI 这类任务常用 BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

        # Optional clustering-enhanced prototype module (disabled by default).
        self.enable_cluster_enhance = bool(enable_cluster_enhance)
        self.intra_clusters = int(intra_clusters)
        self.shared_prototypes_num = int(shared_prototypes_num)
        self.cluster_interval = max(1, int(cluster_interval))
        self.cluster_tau = float(cluster_tau)
        self.lambda_cross = float(lambda_cross)
        self.lambda_reg = float(lambda_reg)
        self.lambda_proto = float(lambda_proto)
        self.cluster_conf_threshold = float(cluster_conf_threshold)
        self.prototype_ema_momentum = float(prototype_ema_momentum)
        self._cluster_step = 0
        self._prev_cluster_ids = {}
        self.last_loss_breakdown = {
            'base_loss': 0.0,
            'cluster_loss': 0.0,
            'loss_proto': 0.0,
            'loss_cross': 0.0,
            'loss_reg': 0.0,
            'num_domains': 0,
            'cluster_scale': 1.0,
            'avg_assignment_entropy': 0.0,
            'avg_min_cluster_ratio': 0.0,
            'avg_max_cluster_ratio': 0.0,
            'avg_cluster_ari': 0.0,
            'cross_gap': 0.0,
            'proto_collapse': 0.0,
            'domain_gap': 0.0,
            'match_coverage': 0.0,
            'cluster_step': 0,
            'cluster_triggered': 0,
        }
        self.last_cluster_stats = []

        self.shared_prototypes = nn.Parameter(torch.randn(self.shared_prototypes_num, n_h) * 0.02)

    def ablation(self, fea_prelogits, str_prelogits):
        """根据 ablation_choice 组合 fea/str 两类 logits."""
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits

    def compute_prelogits_GRAPHCL(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse=False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
    ):
        """计算 GRAPHCL 预训练 logits（逐数据集 yield）。

        seq_list 中的每个 seq 预期是一个 list/stack：
            [feature, shuf_fts, feature.detach(), feature.detach()]
        adj_list 中每个 adj 预期是 stack：
            [adj, aug_adj1, aug_adj2]
        这些格式来自 utils/aug.py 的 build_aug()。

        当 ablation_choice != 'None' 时，会分别计算：
        - fea_prelogits：对 seq 的每个视图做 feature prompt 后喂入 GraphCL
        - str_prelogits：将 structure prompt 传入 gcn，计算结构提示下的 GraphCL
        """
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.graphcledge(
                    self.gcn,
                    seq[0],
                    seq[1],
                    seq[2],
                    seq[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    'edge',
                )
            else:
                # 对每个视图都应用 feature prompt
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))]
                fea_prelogits = self.graphcledge(
                    self.gcn,
                    preseq_list[0],
                    preseq_list[1],
                    preseq_list[2],
                    preseq_list[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    aug_type='edge',
                )

                # 结构 prompt：不改输入 seq，但把 prompt 列表传给 gcn
                str_prelogits = self.graphcledge(
                    self.gcn,
                    seq[0],
                    seq[1],
                    seq[2],
                    seq[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    'edge',
                    str_layers,
                )

                yield self.ablation(fea_prelogits, str_prelogits)

    def _domain_prompted_embedding(self, domain_idx, seq, adj, sparse=False):
        """Build per-domain node embeddings with the same prompt composition used in pretraining."""
        if self.ablation_choice == 'None':
            emb = self.gcn(seq, adj, sparse, None)
            return emb.squeeze(0) if emb.dim() == 3 else emb

        fea_prompt = self.feature_prompt_layers[domain_idx]
        str_prompt = self.structure_prompt_layers[domain_idx]

        fea_emb = self.gcn(fea_prompt(seq), adj, sparse, None)
        str_emb = self.gcn(seq, adj, sparse, None, str_prompt)
        emb = self.ablation(fea_emb, str_emb)
        return emb.squeeze(0) if emb.dim() == 3 else emb

    def _build_domain_embeddings(self, seq_list, adj_list, sparse=False):
        embeds = []
        for domain_idx, (seq_item, adj_item) in enumerate(zip(seq_list, adj_list)):
            # GRAPHCL passes stacked tensors, not Python lists:
            # - seq_item: [4, N, F]
            # - adj_item: [3, N, N]
            # Use the base/original slice at index 0 so the clustering branch
            # sees the same graph as the main encoder.
            seq = seq_item[0] if torch.is_tensor(seq_item) and seq_item.dim() >= 3 else seq_item
            adj = adj_item[0] if torch.is_tensor(adj_item) and adj_item.dim() >= 3 else adj_item
            if torch.is_tensor(adj) and adj.dim() == 3:
                adj = adj[0]
            embeds.append(self._domain_prompted_embedding(domain_idx, seq, adj, sparse))
        return embeds

    def _cluster_domain_centers(self, embedding, seed):
        k = max(1, min(self.intra_clusters, embedding.shape[0]))
        x = F.normalize(embedding.detach(), p=2, dim=1).cpu().numpy()
        clusterer = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
        cluster_ids = clusterer.fit_predict(x)
        centers = torch.as_tensor(clusterer.cluster_centers_, dtype=embedding.dtype, device=embedding.device)
        return cluster_ids, centers

    def _matching_cross_domain_loss(self, domain_centers, domain_assignments):
        """跨域原型匹配后再对齐，避免错误簇强行一一对齐。"""
        if len(domain_centers) < 2:
            zero = torch.tensor(0.0, device=self.shared_prototypes.device)
            return zero, 0.0, 0.0

        pair_losses = []
        cross_gaps = []
        coverage = []

        for i in range(len(domain_centers)):
            ci = F.normalize(domain_centers[i], p=2, dim=1)
            conf_i = torch.max(domain_assignments[i], dim=1).values
            for j in range(i + 1, len(domain_centers)):
                cj = F.normalize(domain_centers[j], p=2, dim=1)
                conf_j = torch.max(domain_assignments[j], dim=1).values

                sim = torch.matmul(ci, cj.T)
                if sim.numel() == 0:
                    continue

                # Hungarian 最大匹配（在相似度矩阵上做最大化）。
                row_idx_np, col_idx_np = linear_sum_assignment((-sim.detach()).cpu().numpy())
                row_idx = torch.as_tensor(row_idx_np, dtype=torch.long, device=sim.device)
                col_idx = torch.as_tensor(col_idx_np, dtype=torch.long, device=sim.device)

                pair_conf = torch.minimum(conf_i[row_idx], conf_j[col_idx])
                valid = pair_conf > self.cluster_conf_threshold
                if torch.sum(valid) == 0:
                    continue

                r_sel = row_idx[valid]
                c_sel = col_idx[valid]
                pos_sim = sim[r_sel, c_sel]

                matched_mask = torch.zeros_like(sim, dtype=torch.bool)
                matched_mask[r_sel, c_sel] = True
                neg_sim = sim[~matched_mask]
                if neg_sim.numel() == 0:
                    neg_sim = sim.reshape(-1)

                pos_mean = torch.mean(pos_sim)
                neg_mean = torch.mean(neg_sim)
                # gap 拉开：匹配相似度应明显高于非匹配相似度。
                gap = pos_mean - neg_mean
                loss_pair = F.relu(0.2 - gap)

                pair_losses.append(loss_pair)
                cross_gaps.append(float(gap.detach().item()))
                coverage.append(float(torch.sum(valid).item() / max(1, len(row_idx_np))))

        if len(pair_losses) == 0:
            zero = torch.tensor(0.0, device=self.shared_prototypes.device)
            return zero, 0.0, 0.0

        return torch.mean(torch.stack(pair_losses)), float(np.mean(cross_gaps)), float(np.mean(coverage))

    def _update_shared_prototypes_ema(self, domain_centers, domain_assignments):
        """用域内中心对共享原型做 EMA 慢更新，减少抖动。"""
        if len(domain_centers) == 0:
            return

        device = self.shared_prototypes.device
        target_sum = torch.zeros_like(self.shared_prototypes.data, device=device)
        weight_sum = torch.zeros((self.shared_prototypes_num, 1), dtype=self.shared_prototypes.dtype, device=device)

        for centers, alpha in zip(domain_centers, domain_assignments):
            centers_n = F.normalize(centers.detach(), p=2, dim=1)
            alpha_n = alpha.detach()
            target_sum = target_sum + torch.matmul(alpha_n.T, centers_n)
            weight_sum = weight_sum + torch.sum(alpha_n, dim=0, keepdim=True).T

        target = target_sum / torch.clamp(weight_sum, min=1e-6)
        target = F.normalize(target, p=2, dim=1)

        m = min(max(self.prototype_ema_momentum, 0.0), 0.999)
        with torch.no_grad():
            self.shared_prototypes.data.mul_(m).add_((1.0 - m) * target)

    def _cluster_enhance_loss(self, seq_list, adj_list, sparse=False):
        domain_embeds = self._build_domain_embeddings(seq_list, adj_list, sparse)
        if len(domain_embeds) == 0:
            self.last_loss_breakdown.update(
                {
                    'cluster_loss': 0.0,
                    'loss_proto': 0.0,
                    'loss_cross': 0.0,
                    'loss_reg': 0.0,
                    'num_domains': 0,
                    'cluster_scale': 1.0,
                    'avg_assignment_entropy': 0.0,
                    'avg_min_cluster_ratio': 0.0,
                    'avg_max_cluster_ratio': 0.0,
                    'avg_cluster_ari': 0.0,
                    'cross_gap': 0.0,
                    'proto_collapse': 0.0,
                    'domain_gap': 0.0,
                    'match_coverage': 0.0,
                }
            )
            self.last_cluster_stats = []
            return torch.tensor(0.0, device=self.shared_prototypes.device)

        shared = F.normalize(self.shared_prototypes, p=2, dim=1)
        domain_cluster_ids = []
        domain_centers = []
        domain_tildes = []
        domain_assignments = []
        domain_stats = []
        ari_scores = []

        for domain_idx, emb in enumerate(domain_embeds):
            cluster_ids_np, centers = self._cluster_domain_centers(emb, seed=domain_idx + self._cluster_step)

            prev_ids = self._prev_cluster_ids.get(domain_idx, None)
            if prev_ids is not None and len(prev_ids) == len(cluster_ids_np):
                try:
                    ari_scores.append(float(adjusted_rand_score(prev_ids, cluster_ids_np)))
                except Exception:
                    pass
            self._prev_cluster_ids[domain_idx] = cluster_ids_np.copy()

            centers_norm = F.normalize(centers, p=2, dim=1)
            sim = torch.matmul(centers_norm, shared.T) / self.cluster_tau
            alpha = F.softmax(sim, dim=-1)
            centers_tilde = torch.matmul(alpha, self.shared_prototypes)
            proto_ids = torch.argmax(alpha, dim=-1)
            cluster_hist = np.bincount(cluster_ids_np, minlength=centers.shape[0]).tolist()
            proto_hist = np.bincount(proto_ids.detach().cpu().numpy(), minlength=self.shared_prototypes_num).tolist()
            hist_sum = float(max(1, sum(cluster_hist)))
            min_cluster_ratio = float(min(cluster_hist) / hist_sum) if len(cluster_hist) > 0 else 0.0
            max_cluster_ratio = float(max(cluster_hist) / hist_sum) if len(cluster_hist) > 0 else 0.0
            assignment_entropy = float((-(alpha * torch.log(alpha + 1e-12)).sum(dim=-1)).mean().detach().item())

            domain_cluster_ids.append(torch.as_tensor(cluster_ids_np, dtype=torch.long, device=emb.device))
            domain_centers.append(centers)
            domain_tildes.append(centers_tilde)
            domain_assignments.append(alpha)
            domain_stats.append(
                {
                    'domain_idx': domain_idx,
                    'num_nodes': int(emb.shape[0]),
                    'num_clusters': int(centers.shape[0]),
                    'num_centers': int(centers.shape[0]),
                    'cluster_hist': cluster_hist,
                    'proto_hist': proto_hist,
                    'min_cluster_ratio': min_cluster_ratio,
                    'max_cluster_ratio': max_cluster_ratio,
                    'assignment_entropy': assignment_entropy,
                }
            )

        domain_node_counts = torch.as_tensor(
            [max(1, int(e.shape[0])) for e in domain_embeds],
            dtype=torch.float32,
            device=domain_embeds[0].device,
        )
        domain_weights = domain_node_counts / torch.clamp(domain_node_counts.sum(), min=1.0)

        # 1) Prototype alignment: domain centers align to shared-reconstructed centers.
        proto_terms = []
        for centers, centers_tilde in zip(domain_centers, domain_tildes):
            proto_terms.append(F.mse_loss(centers, centers_tilde))
        proto_terms = torch.stack(proto_terms)
        loss_proto = torch.sum(domain_weights * proto_terms)

        # 2) Matching-based cross-domain alignment with confidence filtering.
        loss_cross, cross_gap, match_coverage = self._matching_cross_domain_loss(
            domain_centers=domain_centers,
            domain_assignments=domain_assignments,
        )

        # 3) Token-Prototype regularization: each structural token close to nearest local center.
        per_domain_reg = []
        for domain_idx, centers in enumerate(domain_centers):
            if centers.shape[0] == 0:
                per_domain_reg.append(torch.tensor(0.0, device=domain_embeds[0].device))
                continue
            domain_reg = torch.tensor(0.0, device=domain_embeds[0].device)
            token_cnt = 0
            for token_layer in self.structure_prompt_layers[domain_idx]:
                token = token_layer.weight
                token = token.squeeze(0) if token.dim() > 1 else token
                dists = torch.cdist(token.unsqueeze(0), centers)
                nearest_idx = torch.argmin(dists, dim=1)
                nearest_center = centers[nearest_idx].squeeze(0)
                domain_reg = domain_reg + F.mse_loss(token, nearest_center)
                token_cnt += 1
            if token_cnt > 0:
                domain_reg = domain_reg / token_cnt
            per_domain_reg.append(domain_reg)
        per_domain_reg = torch.stack(per_domain_reg)
        loss_reg = torch.sum(domain_weights * per_domain_reg)

        cluster_loss = self.lambda_cross * loss_cross + self.lambda_reg * loss_reg + self.lambda_proto * loss_proto
        # 多域时做温和缩放，避免域数增加后辅助约束过强压制主任务。
        cluster_scale = float(1.0 / np.sqrt(max(1, len(domain_embeds))))
        cluster_loss = cluster_loss * cluster_scale
        self._update_shared_prototypes_ema(domain_centers, domain_assignments)

        proto_norm = F.normalize(self.shared_prototypes, p=2, dim=1)
        proto_sim = torch.matmul(proto_norm, proto_norm.T)
        if proto_sim.shape[0] > 1:
            eye = torch.eye(proto_sim.shape[0], device=proto_sim.device, dtype=torch.bool)
            proto_collapse = float(torch.mean(proto_sim[~eye]).detach().item())
        else:
            proto_collapse = 1.0

        domain_means = torch.stack([F.normalize(torch.mean(e, dim=0, keepdim=False), p=2, dim=0) for e in domain_embeds], dim=0)
        if domain_means.shape[0] > 1:
            gap_mat = torch.cdist(domain_means, domain_means, p=2)
            eye = torch.eye(gap_mat.shape[0], device=gap_mat.device, dtype=torch.bool)
            domain_gap = float(torch.mean(gap_mat[~eye]).detach().item())
        else:
            domain_gap = 0.0

        self.last_cluster_stats = domain_stats
        avg_entropy = float(np.mean([s.get('assignment_entropy', 0.0) for s in domain_stats])) if len(domain_stats) > 0 else 0.0
        avg_min_ratio = float(np.mean([s.get('min_cluster_ratio', 0.0) for s in domain_stats])) if len(domain_stats) > 0 else 0.0
        avg_max_ratio = float(np.mean([s.get('max_cluster_ratio', 0.0) for s in domain_stats])) if len(domain_stats) > 0 else 0.0
        avg_ari = float(np.mean(ari_scores)) if len(ari_scores) > 0 else 0.0
        self.last_loss_breakdown.update(
            {
                'cluster_loss': float(cluster_loss.detach().item()),
                'loss_proto': float(loss_proto.detach().item()),
                'loss_cross': float(loss_cross.detach().item()),
                'loss_reg': float(loss_reg.detach().item()),
                'num_domains': int(len(domain_embeds)),
                'cluster_scale': cluster_scale,
                'avg_assignment_entropy': avg_entropy,
                'avg_min_cluster_ratio': avg_min_ratio,
                'avg_max_cluster_ratio': avg_max_ratio,
                'avg_cluster_ari': avg_ari,
                'cross_gap': cross_gap,
                'proto_collapse': proto_collapse,
                'domain_gap': domain_gap,
                'match_coverage': match_coverage,
                'cluster_step': int(self._cluster_step),
                'cluster_triggered': 1,
            }
        )
        return cluster_loss

    def embed(self, seq, adj, sparse, msk):
        """得到节点 embedding h_1 及图级向量 c（readout）。

        返回 detach 后的张量，通常用于下游评测（不反传梯度）。
        """
        h_1 = self.gcn(seq, adj, sparse, False)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        """导出 prompt 参数（detach）供下游 downprompt 使用."""
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [[layer.weight.detach() for layer in structure_prompt_layer] for structure_prompt_layer in self.structure_prompt_layers]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl):
        """预训练 forward：GRAPHCL 路径。

        Args:
            seq_list/adj_list: 预训练数据列表（多数据集）
            lbl: GRAPHCL 的对比标签（list，与 seq_list 对齐）
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        logits = list(
            self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers,
                self.structure_prompt_layers,
                seq_list,
                adj_list,
                sparse,
                msk,
                samp_bias1,
                samp_bias2,
            )
        )
        for i in range(len(logits)):
            loss = self.loss(logits[i], lbl[i])
            total_loss += loss

        self.last_loss_breakdown['base_loss'] = float(total_loss.detach().item())

        if self.enable_cluster_enhance and (self._cluster_step % self.cluster_interval == 0):
            total_loss = total_loss + self._cluster_enhance_loss(seq_list, adj_list, sparse)
        else:
            self.last_loss_breakdown.update(
                {
                    'cluster_loss': 0.0,
                    'loss_proto': 0.0,
                    'loss_cross': 0.0,
                    'loss_reg': 0.0,
                    'num_domains': len(seq_list),
                    'cluster_scale': 1.0,
                    'avg_assignment_entropy': 0.0,
                    'avg_min_cluster_ratio': 0.0,
                    'avg_max_cluster_ratio': 0.0,
                    'avg_cluster_ari': 0.0,
                    'cross_gap': 0.0,
                    'proto_collapse': 0.0,
                    'domain_gap': 0.0,
                    'match_coverage': 0.0,
                    'cluster_step': int(self._cluster_step),
                    'cluster_triggered': 0,
                }
            )
        self._cluster_step += 1

        self.last_loss_breakdown['total_loss'] = float(total_loss.detach().item())

        return total_loss



def pca_compression(seq, k):
    """PCA 降维到 k 维，并打印累计解释方差比例."""
    seq = np.asarray(seq)
    n_samples, n_features = seq.shape
    use_k = int(min(k, n_samples, n_features))
    if use_k <= 0:
        raise ValueError(f'Invalid PCA target dim: requested {k}, data shape {seq.shape}')
    if use_k != k:
        print(f'Warning: reduce PCA components from {k} to {use_k} because of data shape {seq.shape}')

    pca = PCA(n_components=use_k)
    seq = pca.fit_transform(seq)

    print(pca.explained_variance_ratio_.sum())
    return seq


def svd_compression(seq, k):
    """SVD 压缩（保留前 k 个奇异值）。"""
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:, :k].shape)
    print(VT[:k, :].shape)
    res = U[:, :k].dot(np.diag(Sigma[:k]))

    return res


