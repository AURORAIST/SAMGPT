from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers
from layers import AvgReadout 
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from layers.prompt import *
import copy

# ------------------------------------------------------------
# preprompt.py：预训练阶段的核心模型 PrePrompt。
#
# 该模块的要点：
# 1) 预训练任务支持两类：
#    - GRAPHCL：需要两份增强视图 + DGI/GraphCL 类对比损失（BCEWithLogitsLoss）
#    - LP / splitLP：基于负采样 tuples 的对比损失 compareloss（自定义）
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
        egraphcl_temperature=0.2,
        egraphcl_pair_chunk_size=256,
        egraphcl_candidate_pool=1024,
        egraphcl_soft_topk=10,
        egraphcl_max_negatives=512,
        egraphcl_mask_source='structure',
        egraphcl_online_tau=0.85,
        egraphcl_adaptive_tau=1,
        egraphcl_target_mask_ratio=0.35,
        egraphcl_tau_min=0.80,
        egraphcl_tau_max=0.99,
        egraphcl_batch_size=128,
        egraphcl_min_negatives_per_anchor=1,
        egraphcl_structure_min_negatives_per_anchor=8,
        egraphcl_max_mask_ratio=0.95,
        egraphcl_curve_bins=200,
        egraphcl_curve_chunk_rows=512,
    ):
        super(PrePrompt, self).__init__()

        # 两类预训练目标的封装
        self.lp = Lp(n_in, n_h)
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
        self.egraphcl_proj = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.ReLU(inplace=True),
            nn.Linear(n_h, n_h),
        )
        self.egraphcl_temperature = float(egraphcl_temperature)
        self.egraphcl_pair_chunk_size = int(egraphcl_pair_chunk_size)
        self.egraphcl_candidate_pool = int(egraphcl_candidate_pool)
        self.egraphcl_soft_topk = int(egraphcl_soft_topk)
        self.egraphcl_max_negatives = int(egraphcl_max_negatives)
        # Legacy config: keep parser compatibility, but single-branch EGRAPHCL
        # now always builds soft mask from encoded sample similarity.
        self.egraphcl_mask_source = str(egraphcl_mask_source).lower().strip()
        self.egraphcl_online_tau = float(egraphcl_online_tau)
        self.egraphcl_adaptive_tau = bool(int(egraphcl_adaptive_tau))
        self.egraphcl_target_mask_ratio = float(np.clip(egraphcl_target_mask_ratio, 0.01, 0.95))
        self.egraphcl_tau_min = float(np.clip(egraphcl_tau_min, 0.0, 1.0))
        self.egraphcl_tau_max = float(np.clip(egraphcl_tau_max, 0.0, 1.0))
        if self.egraphcl_tau_min > self.egraphcl_tau_max:
            self.egraphcl_tau_min, self.egraphcl_tau_max = self.egraphcl_tau_max, self.egraphcl_tau_min
        self.egraphcl_batch_size = int(egraphcl_batch_size)
        self.egraphcl_min_negatives_per_anchor = max(1, int(egraphcl_min_negatives_per_anchor))
        self.egraphcl_structure_min_negatives_per_anchor = max(1, int(egraphcl_structure_min_negatives_per_anchor))
        self.egraphcl_max_mask_ratio = float(np.clip(egraphcl_max_mask_ratio, 0.0, 0.999))
        self.egraphcl_curve_bins = max(32, int(egraphcl_curve_bins))
        self.egraphcl_curve_chunk_rows = max(64, int(egraphcl_curve_chunk_rows))
        self.egraphcl_collect_curve = False
        self.last_egraphcl_similarity_curve = {}
        self.last_egraphcl_stats = {
            'masked_pairs': 0,
            'total_pairs': 0,
            'active_negative_pairs': 0,
            'mask_ratio': 0.0,
            'tau_used': float(self.egraphcl_online_tau),
            'batch_size': int(egraphcl_batch_size),
            'num_batches': 0,
        }

        self.ablation_choice = ablation

    @staticmethod
    def masked_bce_with_logits(logits, target, bce_mask=None):
        """BCE loss with ignore label support.

        bce_mask==0 的位置不参与损失（用于软正例屏蔽）。
        """
        if bce_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        else:
            valid_mask = bce_mask > 0
        if torch.sum(valid_mask).item() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        per_elem = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return per_elem[valid_mask].mean()

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

    def compute_prelogits_LP(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse=False,
        sample_dataset_ids=None,
    ):
        """计算 LP/splitLP 预训练的 logits（逐数据集 yield）。

        对每个预训练数据集：
        - fea_prelogits：先对 seq 做 feature prompt，再做 LP 任务
        - str_prelogits：把 structure prompt 传入 gcn（影响每层），再做 LP 任务
        - 最终通过 ablation() 得到融合 logits
        """
        if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
            sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()

        for sample_i, (seq, adj) in enumerate(zip(seq_list, adj_list)):
            dataset_id = sample_i if sample_dataset_ids is None else int(sample_dataset_ids[sample_i])
            if dataset_id < 0 or dataset_id >= len(feature_prompt_layers):
                raise IndexError(
                    f"dataset_id={dataset_id} out of range for prompt layers (size={len(feature_prompt_layers)})"
                )
            fea_pretext = feature_prompt_layers[dataset_id]
            str_layers = structure_prompt_layers[dataset_id]
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)

    def compute_prelogits_GRAPHCL(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        pretrain_method='GRAPHCL',
        sparse=False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
        sample_dataset_ids=None,
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
        # Keep a single GraphCL head; EGRAPHCL differences come from labels/masking.
        graphcl_runner = self.graphcledge

        if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
            sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()

        for sample_i, (seq, adj) in enumerate(zip(seq_list, adj_list)):
            dataset_id = sample_i if sample_dataset_ids is None else int(sample_dataset_ids[sample_i])
            if dataset_id < 0 or dataset_id >= len(feature_prompt_layers):
                raise IndexError(
                    f"dataset_id={dataset_id} out of range for prompt layers (size={len(feature_prompt_layers)})"
                )
            fea_pretext = feature_prompt_layers[dataset_id]
            str_layers = structure_prompt_layers[dataset_id]
            if self.ablation_choice == 'None':
                yield graphcl_runner(
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
                fea_prelogits = graphcl_runner(
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
                str_prelogits = graphcl_runner(
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

    def embed(self, seq, adj, sparse, msk, LP):
        """得到节点 embedding h_1 及图级向量 c（readout）。

        返回 detach 后的张量，通常用于下游评测（不反传梯度）。
        """
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def _compute_graph_pair_vectors(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse=False,
        sample_dataset_ids=None,
        use_feature_prompt=True,
        use_structure_prompt=False,
    ):
        """Build sample-level graph embeddings for two augmented graph views.

        This implementation batches multiple subgraphs by dataset id and uses
        block-diagonal adjacency so one GNN forward handles many graphs.
        """
        if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
            sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()

        def _as_sparse_2d(adj_tensor):
            if adj_tensor.layout == torch.sparse_coo:
                return adj_tensor.coalesce()
            return adj_tensor.to_sparse().coalesce()

        def _build_block_diag_adj(adj_parts, use_sparse, device):
            if len(adj_parts) == 1:
                one = adj_parts[0].to(device, non_blocking=True)
                return _as_sparse_2d(one) if use_sparse else one.to_dense() if one.layout == torch.sparse_coo else one

            if use_sparse:
                indices_all = []
                values_all = []
                offset = 0
                for a in adj_parts:
                    a_dev = _as_sparse_2d(a.to(device, non_blocking=True))
                    idx = a_dev.indices()
                    val = a_dev.values()
                    shifted = idx + offset
                    indices_all.append(shifted)
                    values_all.append(val)
                    offset += int(a_dev.size(0))
                indices = torch.cat(indices_all, dim=1)
                values = torch.cat(values_all, dim=0)
                size = torch.Size([offset, offset])
                return torch.sparse_coo_tensor(indices, values, size=size, device=device).coalesce()

            dense_parts = []
            for a in adj_parts:
                a_dev = a.to(device, non_blocking=True)
                if a_dev.layout == torch.sparse_coo:
                    a_dev = a_dev.to_dense()
                dense_parts.append(a_dev)
            return torch.block_diag(*dense_parts)

        def _graph_mean_pool(node_repr, node_sizes):
            segs = torch.split(node_repr, node_sizes, dim=0)
            return torch.stack([seg.mean(dim=0) for seg in segs], dim=0)

        n_samples = len(seq_list)
        device = next(self.parameters()).device
        batch_graphs = max(2, int(self.egraphcl_batch_size))

        if sample_dataset_ids is None:
            sample_dataset_ids = list(range(n_samples))

        by_dataset = {}
        for i, ds in enumerate(sample_dataset_ids):
            ds_id = int(ds)
            by_dataset.setdefault(ds_id, []).append(i)

        z1_out = [None] * n_samples
        z2_out = [None] * n_samples

        for dataset_id, sample_indices in by_dataset.items():
            if dataset_id < 0 or dataset_id >= len(feature_prompt_layers):
                raise IndexError(
                    f"dataset_id={dataset_id} out of range for prompt layers (size={len(feature_prompt_layers)})"
                )

            fea_pretext = feature_prompt_layers[dataset_id]
            str_layers = structure_prompt_layers[dataset_id]

            for start in range(0, len(sample_indices), batch_graphs):
                end = min(start + batch_graphs, len(sample_indices))
                b_idx = sample_indices[start:end]
                if len(b_idx) == 0:
                    continue

                seq_parts = [seq_list[i][0].to(device, non_blocking=True) for i in b_idx]
                adj1_parts = [adj_list[i][1] for i in b_idx]
                adj2_parts = [adj_list[i][2] for i in b_idx]
                node_sizes = [int(t.size(0)) for t in seq_parts]

                seq_cat = torch.cat(seq_parts, dim=0)
                if use_feature_prompt:
                    seq_cat = fea_pretext(seq_cat)
                prompt_layers = str_layers if use_structure_prompt else None

                adj1_block = _build_block_diag_adj(adj1_parts, sparse, device)
                adj2_block = _build_block_diag_adj(adj2_parts, sparse, device)

                h1_cat = self.gcn(seq_cat, adj1_block, sparse, prompt_layers=prompt_layers).squeeze(0)
                h2_cat = self.gcn(seq_cat, adj2_block, sparse, prompt_layers=prompt_layers).squeeze(0)
                c1_batch = _graph_mean_pool(h1_cat, node_sizes)
                c2_batch = _graph_mean_pool(h2_cat, node_sizes)

                for local_i, sample_i in enumerate(b_idx):
                    z1_out[sample_i] = c1_batch[local_i]
                    z2_out[sample_i] = c2_batch[local_i]

        z1 = torch.stack(z1_out, dim=0)
        z2 = torch.stack(z2_out, dim=0)
        z1 = self.egraphcl_proj(z1)
        z2 = self.egraphcl_proj(z2)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return z1, z2

    def _build_local_online_soft_mask(self, z1_local, z2_local, min_negatives_per_anchor=1):
        """Build soft-positive mask only within a mini-batch to avoid NxN allocation."""
        bsz = int(z1_local.size(0))
        if bsz <= 1:
            return torch.zeros((bsz, bsz), device=z1_local.device, dtype=torch.bool)

        # Single-view reference for online masking: use encoded view-1 directly.
        z_ref = F.normalize(z1_local, p=2, dim=1)
        sim = torch.matmul(z_ref, z_ref.t())
        tau_used = float(self.egraphcl_online_tau)
        if self.egraphcl_adaptive_tau and bsz > 2:
            eye = torch.eye(bsz, device=sim.device, dtype=torch.bool)
            sim_offdiag = sim[~eye]
            if sim_offdiag.numel() > 0:
                q = float(1.0 - self.egraphcl_target_mask_ratio)
                q = min(max(q, 0.0), 1.0)
                tau_q = float(torch.quantile(sim_offdiag, q).item())
                tau_used = float(np.clip(tau_q, self.egraphcl_tau_min, self.egraphcl_tau_max))

        soft_mask = sim >= float(tau_used)
        eye = torch.eye(bsz, device=soft_mask.device, dtype=torch.bool)
        soft_mask = soft_mask & (~eye)

        max_mask_per_anchor = int(np.floor((bsz - 1) * self.egraphcl_max_mask_ratio))
        min_neg = max(1, min(int(min_negatives_per_anchor), bsz - 1))
        max_mask_per_anchor = min(max_mask_per_anchor, (bsz - 1) - min_neg)
        max_mask_per_anchor = max(0, max_mask_per_anchor)

        for i in range(bsz):
            masked_idx = torch.where(soft_mask[i])[0]
            excess = int(masked_idx.numel()) - max_mask_per_anchor
            if excess <= 0:
                continue
            masked_sim = sim[i, masked_idx]
            drop_local = torch.topk(masked_sim, k=excess, largest=False).indices
            drop_idx = masked_idx[drop_local]
            soft_mask[i, drop_idx] = False

        soft_mask = soft_mask & soft_mask.t()
        soft_mask = soft_mask & (~eye)
        self.last_egraphcl_stats['tau_used'] = float(tau_used)
        return soft_mask

    def _record_similarity_curve_chunked(self, z1, z2, curve_tag='feature'):
        """Collect full-set similarity histogram in row chunks without materializing NxN."""
        if not bool(getattr(self, 'egraphcl_collect_curve', False)):
            return

        # Single-view reference for curve stats: use encoded view-1 directly.
        z_ref = F.normalize(z1, p=2, dim=1)
        n = int(z_ref.size(0))
        if n <= 1:
            return

        bins = int(self.egraphcl_curve_bins)
        row_chunk = int(self.egraphcl_curve_chunk_rows)
        counts = np.zeros((bins,), dtype=np.float64)
        edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float64)

        z_ref_t = z_ref.t()
        for start in range(0, n, row_chunk):
            end = min(start + row_chunk, n)
            sim_chunk = torch.matmul(z_ref[start:end], z_ref_t)
            chunk = sim_chunk.detach().float().cpu().numpy()
            local_mask = np.ones_like(chunk, dtype=bool)
            for i_local, i_global in enumerate(range(start, end)):
                local_mask[i_local, i_global] = False
            vals = chunk[local_mask]
            hist, _ = np.histogram(vals, bins=edges)
            counts += hist.astype(np.float64)

        total = float(max(1.0, counts.sum()))
        cdf = np.cumsum(counts) / total
        centers = (edges[:-1] + edges[1:]) * 0.5

        def _interp_quantile(target_q):
            idx = int(np.searchsorted(cdf, float(target_q), side='left'))
            idx = min(max(idx, 0), len(centers) - 1)
            return float(centers[idx])

        self.last_egraphcl_similarity_curve[str(curve_tag)] = {
            'bin_centers': centers.tolist(),
            'density': (counts / total).tolist(),
            'cdf': cdf.tolist(),
            'q25': _interp_quantile(0.25),
            'q50': _interp_quantile(0.50),
            'q75': _interp_quantile(0.75),
            'num_pairs': int(total),
        }

    def _sample_pair_contrastive_loss_minibatch(
        self,
        z1,
        z2,
        soft_pair_mask,
        perm=None,
        online_mask_embeddings=None,
        min_negatives_per_anchor=1,
    ):
        """Mini-batch sample-level contrastive loss with in-batch soft-positive masking."""
        bsz = z1.size(0)
        if bsz <= 1:
            self.last_egraphcl_stats = {
                'masked_pairs': 0,
                'total_pairs': 0,
                'active_negative_pairs': 0,
                'mask_ratio': 0.0,
                'batch_size': int(self.egraphcl_batch_size),
                'num_batches': 0,
            }
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

        if soft_pair_mask is not None:
            soft_pair_mask = soft_pair_mask.to(z1.device) > 0

        temperature = max(1e-6, float(self.egraphcl_temperature))
        batch_size = max(2, int(self.egraphcl_batch_size))
        if perm is None:
            perm = torch.randperm(bsz, device=z1.device)

        total_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        total_count = 0
        masked_pairs = 0
        total_pairs = 0
        active_negative_pairs = 0
        num_batches = 0

        for start in range(0, bsz, batch_size):
            end = min(start + batch_size, bsz)
            b_idx = perm[start:end]
            if b_idx.numel() <= 1:
                continue
            num_batches += 1

            zb1 = z1[b_idx]
            zb2 = z2[b_idx]
            logits = torch.matmul(zb1, zb2.t()) / temperature
            exp_logits = torch.exp(logits)

            eye = torch.eye(b_idx.numel(), device=z1.device, dtype=torch.bool)
            neg_valid = ~eye

            if soft_pair_mask is not None:
                soft_local = soft_pair_mask[b_idx][:, b_idx] & (~eye)
            elif online_mask_embeddings is not None:
                mask_z1, mask_z2 = online_mask_embeddings
                soft_local = self._build_local_online_soft_mask(
                    mask_z1[b_idx],
                    mask_z2[b_idx],
                    min_negatives_per_anchor=min_negatives_per_anchor,
                )
                soft_local = soft_local & (~eye)
            else:
                soft_local = torch.zeros_like(eye)
            neg_valid = neg_valid & (~soft_local)

            total_pairs += int((~eye).sum().item())
            masked_pairs += int(soft_local.sum().item())

            # Hard negative cap (within batch)
            max_negs = int(self.egraphcl_max_negatives)
            if max_negs > 0:
                for i in range(neg_valid.size(0)):
                    cols = torch.where(neg_valid[i])[0]
                    if cols.numel() <= max_negs:
                        continue
                    vals = logits[i, cols]
                    keep_cols = cols[torch.topk(vals, k=max_negs, largest=True).indices]
                    keep = torch.zeros_like(neg_valid[i], dtype=torch.bool)
                    keep[keep_cols] = True
                    neg_valid[i] = keep

            active_negative_pairs += int(neg_valid.sum().item())

            pos = torch.exp(torch.diag(logits))
            neg = (exp_logits * neg_valid.float()).sum(dim=1)
            denom = pos + neg
            valid = denom > 0
            if torch.any(valid):
                loss_vec = -torch.log((pos[valid] + 1e-12) / (denom[valid] + 1e-12))
                total_loss = total_loss + loss_vec.sum()
                total_count += int(valid.sum().item())

        mask_ratio = float(masked_pairs / max(1, total_pairs))
        self.last_egraphcl_stats = {
            'masked_pairs': int(masked_pairs),
            'total_pairs': int(total_pairs),
            'active_negative_pairs': int(active_negative_pairs),
            'mask_ratio': float(mask_ratio),
            'batch_size': int(batch_size),
            'num_batches': int(num_batches),
        }

        if total_count == 0:
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        return total_loss / float(total_count)

    def _stream_egraphcl_loss(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse=False,
        sample_dataset_ids=None,
        use_feature_prompt=True,
        use_structure_prompt=True,
        min_negatives_per_anchor=1,
    ):
        """Compute EGRAPHCL loss in streaming chunks to avoid retaining full-epoch autograd graph."""
        if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
            sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()

        def _as_sparse_2d(adj_tensor):
            if adj_tensor.layout == torch.sparse_coo:
                return adj_tensor.coalesce()
            return adj_tensor.to_sparse().coalesce()

        def _build_block_diag_adj(adj_parts, use_sparse, dev):
            if len(adj_parts) == 1:
                one = adj_parts[0].to(dev, non_blocking=True)
                return _as_sparse_2d(one) if use_sparse else one.to_dense() if one.layout == torch.sparse_coo else one

            if use_sparse:
                indices_all = []
                values_all = []
                offset = 0
                for a in adj_parts:
                    a_dev = _as_sparse_2d(a.to(dev, non_blocking=True))
                    idx = a_dev.indices()
                    val = a_dev.values()
                    shifted = idx + offset
                    indices_all.append(shifted)
                    values_all.append(val)
                    offset += int(a_dev.size(0))
                indices = torch.cat(indices_all, dim=1)
                values = torch.cat(values_all, dim=0)
                size = torch.Size([offset, offset])
                return torch.sparse_coo_tensor(indices, values, size=size, device=dev).coalesce()

            dense_parts = []
            for a in adj_parts:
                a_dev = a.to(dev, non_blocking=True)
                if a_dev.layout == torch.sparse_coo:
                    a_dev = a_dev.to_dense()
                dense_parts.append(a_dev)
            return torch.block_diag(*dense_parts)

        def _graph_mean_pool(node_repr, node_sizes):
            segs = torch.split(node_repr, node_sizes, dim=0)
            return torch.stack([seg.mean(dim=0) for seg in segs], dim=0)

        n_samples = len(seq_list)
        if n_samples <= 1:
            self.last_egraphcl_stats = {
                'masked_pairs': 0,
                'total_pairs': 0,
                'active_negative_pairs': 0,
                'mask_ratio': 0.0,
                'batch_size': int(self.egraphcl_batch_size),
                'num_batches': 0,
            }
            return torch.tensor(0.0, device=next(self.parameters()).device)

        dev = next(self.parameters()).device
        batch_graphs = max(2, int(self.egraphcl_batch_size))
        temperature = max(1e-6, float(self.egraphcl_temperature))

        if sample_dataset_ids is None:
            sample_dataset_ids = list(range(n_samples))

        by_dataset = {}
        for i, ds in enumerate(sample_dataset_ids):
            ds_id = int(ds)
            by_dataset.setdefault(ds_id, []).append(i)

        total_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        total_count = 0
        masked_pairs = 0
        total_pairs = 0
        active_negative_pairs = 0
        num_batches = 0

        for dataset_id, sample_indices in by_dataset.items():
            if dataset_id < 0 or dataset_id >= len(feature_prompt_layers):
                raise IndexError(
                    f"dataset_id={dataset_id} out of range for prompt layers (size={len(feature_prompt_layers)})"
                )

            fea_pretext = feature_prompt_layers[dataset_id]
            str_layers = structure_prompt_layers[dataset_id]

            for start in range(0, len(sample_indices), batch_graphs):
                end = min(start + batch_graphs, len(sample_indices))
                b_idx = sample_indices[start:end]
                if len(b_idx) <= 1:
                    continue
                num_batches += 1

                seq_parts = [seq_list[i][0].to(dev, non_blocking=True) for i in b_idx]
                adj1_parts = [adj_list[i][1] for i in b_idx]
                adj2_parts = [adj_list[i][2] for i in b_idx]
                node_sizes = [int(t.size(0)) for t in seq_parts]

                seq_cat = torch.cat(seq_parts, dim=0)
                if use_feature_prompt:
                    seq_cat = fea_pretext(seq_cat)
                prompt_layers = str_layers if use_structure_prompt else None

                adj1_block = _build_block_diag_adj(adj1_parts, sparse, dev)
                adj2_block = _build_block_diag_adj(adj2_parts, sparse, dev)

                h1_cat = self.gcn(seq_cat, adj1_block, sparse, prompt_layers=prompt_layers).squeeze(0)
                h2_cat = self.gcn(seq_cat, adj2_block, sparse, prompt_layers=prompt_layers).squeeze(0)
                z1 = _graph_mean_pool(h1_cat, node_sizes)
                z2 = _graph_mean_pool(h2_cat, node_sizes)

                z1 = self.egraphcl_proj(z1)
                z2 = self.egraphcl_proj(z2)
                z1 = F.normalize(z1, p=2, dim=1)
                z2 = F.normalize(z2, p=2, dim=1)

                logits = torch.matmul(z1, z2.t()) / temperature
                exp_logits = torch.exp(logits)

                bsz = int(z1.size(0))
                eye = torch.eye(bsz, device=dev, dtype=torch.bool)
                soft_local = self._build_local_online_soft_mask(
                    z1,
                    z2,
                    min_negatives_per_anchor=min_negatives_per_anchor,
                ) & (~eye)
                neg_valid = (~eye) & (~soft_local)

                total_pairs += int((~eye).sum().item())
                masked_pairs += int(soft_local.sum().item())

                max_negs = int(self.egraphcl_max_negatives)
                if max_negs > 0:
                    for i in range(neg_valid.size(0)):
                        cols = torch.where(neg_valid[i])[0]
                        if cols.numel() <= max_negs:
                            continue
                        vals = logits[i, cols]
                        keep_cols = cols[torch.topk(vals, k=max_negs, largest=True).indices]
                        keep = torch.zeros_like(neg_valid[i], dtype=torch.bool)
                        keep[keep_cols] = True
                        neg_valid[i] = keep

                active_negative_pairs += int(neg_valid.sum().item())

                pos = torch.exp(torch.diag(logits))
                neg = (exp_logits * neg_valid.float()).sum(dim=1)
                denom = pos + neg
                valid = denom > 0
                if torch.any(valid):
                    loss_vec = -torch.log((pos[valid] + 1e-12) / (denom[valid] + 1e-12))
                    total_loss = total_loss + loss_vec.sum()
                    total_count += int(valid.sum().item())

        mask_ratio = float(masked_pairs / max(1, total_pairs))
        self.last_egraphcl_stats = {
            'masked_pairs': int(masked_pairs),
            'total_pairs': int(total_pairs),
            'active_negative_pairs': int(active_negative_pairs),
            'mask_ratio': float(mask_ratio),
            'batch_size': int(batch_graphs),
            'num_batches': int(num_batches),
        }

        if total_count == 0:
            return torch.tensor(0.0, device=dev, dtype=torch.float32)
        return total_loss / float(total_count)

    def get_weights(self):
        """导出 prompt 参数（detach）供下游 downprompt 使用."""
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [[layer.weight.detach() for layer in structure_prompt_layer] for structure_prompt_layer in self.structure_prompt_layers]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def forward(
        self,
        seq_list,
        adj_list,
        sparse,
        msk,
        samp_bias1,
        samp_bias2,
        lbl,
        bce_mask=None,
        samples=None,
        pretrain_method='GRAPHCL',
        sample_dataset_ids=None,
    ):
        """预训练 forward：根据是否提供 samples 来区分 GRAPHCL 与 LP。

        Args:
            seq_list/adj_list: 预训练数据列表（多数据集）
            lbl: GRAPHCL 的对比标签（list，与 seq_list 对齐）
            samples: LP/splitLP 的负采样 tuples
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        if samples is None and pretrain_method == 'EGRAPHCL':
            total_loss = self._stream_egraphcl_loss(
                self.feature_prompt_layers,
                self.structure_prompt_layers,
                seq_list,
                adj_list,
                sparse=sparse,
                sample_dataset_ids=sample_dataset_ids,
                use_feature_prompt=False,
                use_structure_prompt=True,
                min_negatives_per_anchor=self.egraphcl_min_negatives_per_anchor,
            )
            stats = dict(self.last_egraphcl_stats)
            stats['mask_source'] = 'encoded_similarity'
            stats['branch'] = 'single_streaming'
            self.last_egraphcl_stats = stats
            return total_loss

        # GRAPHCL 路径：samples==None
        if samples == None:
            logits = list(
                self.compute_prelogits_GRAPHCL(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    pretrain_method=pretrain_method,
                    sparse=sparse,
                    msk=msk,
                    samp_bias1=samp_bias1,
                    samp_bias2=samp_bias2,
                    sample_dataset_ids=sample_dataset_ids,
                )
            )
            for i in range(len(logits)):
                mask_i = None if bce_mask is None else bce_mask[i]
                loss = self.masked_bce_with_logits(logits[i], lbl[i], mask_i)
                total_loss += loss

        # LP / splitLP 路径
        else:
            logits = list(
                self.compute_prelogits_LP(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    sparse,
                    sample_dataset_ids=sample_dataset_ids,
                )
            )

            # splitLP：samples 是 list（每个数据集各自一份 tuples）
            if type(samples) == list:
                samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device) for sample in samples]
                for i in range(len(logits)):
                    loss = compareloss(logits[i], samples[i], temperature=1)
                    total_loss += loss

            # LP：samples 是一个整体 tuples（来自合并图）
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)

        return total_loss



def pca_compression(seq, k):
    """PCA 降维到 k 维，并打印累计解释方差比例."""
    pca = PCA(n_components=k)
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


def mygather(feature, index):
    """辅助函数：按 index 从 feature 中 gather，并保持 batch 结构。

    Args:
        feature: Tensor [N, F]
        index: Tensor [B, K]（或可 reshape 成类似结构）

    Returns:
        Tensor [B, K, F]
    """
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    """一个基于 cosine similarity 的对比损失（与 prompt_pretrain_sample 生成的 tuples 配套）。

    tuples 的每行：
        [pos, neg1, neg2, ...]
    其中：
        - h_i: anchor（按行号 i 取 feature[i]）
        - h_tuples: 从 tuples 中 gather 的 pos/neg 表示
    损失形式接近 InfoNCE：
        -log( exp(sim(anchor,pos))/sum exp(sim(anchor,neg_j)) )

    注意：实现里 exp(sim)/temperature 的写法与常见 exp(sim/temperature) 不同，保持原样不改。
    """
    h_tuples = mygather(feature, tuples)

    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)

    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj, n):
    """为 LP/splitLP 生成负采样 tuples。

    对每个节点 i：
    - 从其非零邻居中随机取 1 个作为“正样本”(pos)
      若无邻居则 pos=i
    - 从其零邻居（非相连节点）中随机取 n 个作为负样本

    Returns:
        Tensor [N, 1+n]，每行 [pos, neg1..negn]

    要求：adj 为 scipy.sparse.csr_matrix（使用 indices/indptr）。
    """
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))

    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i] : indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1 : 1 + n] = zero_index_i_row[0:n]

    return torch.tensor(res.astype(int))