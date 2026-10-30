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
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
import os


logger = logging.getLogger(__name__)

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
        domain_align_clusters=0,
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

        self.ablation_choice = ablation

        self.domain_align_clusters = domain_align_clusters
        if domain_align_clusters > 0:
            self.domain_prototypes = nn.Parameter(torch.FloatTensor(domain_align_clusters, n_h))
            nn.init.xavier_uniform_(self.domain_prototypes)
        else:
            self.register_parameter('domain_prototypes', None)

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

    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, sparse=False):
        """计算 LP/splitLP 预训练的 logits（逐数据集 yield）。

        对每个预训练数据集：
        - fea_prelogits：先对 seq 做 feature prompt，再做 LP 任务
        - str_prelogits：把 structure prompt 传入 gcn（影响每层），再做 LP 任务
        - 最终通过 ablation() 得到融合 logits
        """
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
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
        sparse=False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
        graphcl_node_indices=None,
        graphcl_negative_indices=None,
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
        for domain_id, (fea_pretext, str_layers, seq, adj) in enumerate(zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list)):
            node_idx = None if graphcl_node_indices is None else graphcl_node_indices[domain_id]
            neg_idx = None if graphcl_negative_indices is None else graphcl_negative_indices[domain_id]
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
                    node_idx=node_idx,
                    neg_idx=neg_idx,
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
                    node_idx=node_idx,
                    neg_idx=neg_idx,
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
                    prompt_layers=str_layers,
                    node_idx=node_idx,
                    neg_idx=neg_idx,
                )

                yield self.ablation(fea_prelogits, str_prelogits)


    def _negclip_pair_loss(self, anchor_proto, positive_proto, hard_negative_proto, temperature):
        """NegCLIP-style hard-negative NCE for one prototype pair set.

        ``anchor_proto[k]`` should match ``positive_proto[k]``. The denominator
        contains all positive prototypes plus explicit hard negatives, but the
        reverse direction is kept as the standard positive-only CLIP CE term for
        stability, following the NegCLIP idea of adding hard negatives to the
        retrieval direction where they are reliable.
        """
        proto_num = anchor_proto.shape[0]
        target = torch.arange(proto_num, device=anchor_proto.device)

        pos_logits = torch.matmul(anchor_proto, positive_proto.t()) / temperature
        pos_ce = F.cross_entropy(pos_logits, target)

        neg_logits = torch.matmul(anchor_proto, hard_negative_proto.t()) / temperature
        neg_logits = neg_logits.view(proto_num, -1, proto_num)
        eye_mask = torch.eye(proto_num, dtype=torch.bool, device=anchor_proto.device).unsqueeze(0)
        neg_logits = neg_logits.masked_fill(eye_mask, -1e9)
        neg_logits = neg_logits.permute(1, 0, 2).reshape(proto_num, -1)

        positive_score = pos_logits.diag().unsqueeze(1)
        hard_nce_logits = torch.cat((positive_score, neg_logits), dim=1)
        hard_nce_targets = torch.zeros(proto_num, dtype=torch.long, device=anchor_proto.device)
        hard_negative_ce = F.cross_entropy(hard_nce_logits, hard_nce_targets)

        return 0.5 * (hard_negative_ce + pos_ce)

    def compute_domain_shared_prototype_loss(
        self,
        seq,
        adj,
        sparse,
        node_idx=None,
        temperature=0.2,
        balance_weight=0.05,
    ):
        """Memory-bounded NegCLIP alignment for one domain against shared prototypes."""
        if self.domain_prototypes is None or self.domain_align_clusters <= 0:
            return torch.tensor(0.0, dtype=torch.float32).to(seq.device)

        eps = 1e-8
        temperature = max(float(temperature), eps)
        shared_proto = F.normalize(self.domain_prototypes, dim=-1)

        h = self.gcn(seq[0], adj[0], sparse)
        if h.dim() == 3:
            h = h.squeeze(0)
        if node_idx is not None:
            h = h[node_idx]
        h = F.normalize(h, dim=-1)

        assign_logits = torch.matmul(h, shared_proto.t()) / temperature
        assign_probs = F.softmax(assign_logits, dim=-1)
        with torch.no_grad():
            max_prob = assign_probs.max(dim=1).values.mean().item()
            entropy = (-assign_probs * torch.log(assign_probs + eps)).sum(dim=1).mean().item()
            usage = assign_probs.mean(dim=0)
            usage_min = usage.min().item()
            usage_max = usage.max().item()
            logger.debug(
                'soft-assign single-domain: nodes=%d max_prob=%.4f entropy=%.4f usage[min=%.4f max=%.4f]',
                assign_probs.shape[0],
                max_prob,
                entropy,
                usage_min,
                usage_max,
            )
        cluster_mass = assign_probs.sum(dim=0).clamp_min(eps).unsqueeze(-1)
        local_proto = torch.matmul(assign_probs.t(), h) / cluster_mass
        local_proto = F.normalize(local_proto, dim=-1)

        usage = assign_probs.mean(dim=0).clamp_min(eps)
        usage_loss = torch.sum(usage * torch.log(usage * self.domain_align_clusters))
        return self._negclip_pair_loss(local_proto, shared_proto, shared_proto, temperature) + balance_weight * usage_loss

    def compute_domain_prototype_loss(
        self,
        seq_list,
        adj_list,
        sparse,
        node_indices=None,
        temperature=0.2,
        balance_weight=0.05,
    ):
        """Online cross-domain NegCLIP loss over shared prototypes.

        Static KMeans only bootstraps a high-quality candidate pool. During
        training, current GNN embeddings are softly assigned to learnable shared
        prototypes and each domain recomputes local prototypes online. For each
        local-domain prototype id, its positive is the same shared prototype id;
        hard negatives are the other prototype ids from the shared bank and the
        other domains. This follows NegCLIP's key idea: add targeted hard
        negatives to the anchor-to-target denominator while keeping the reverse
        positive retrieval direction stable. A small usage regularizer prevents
        assigning all nodes to a single prototype.
        """
        if self.domain_prototypes is None or self.domain_align_clusters <= 0:
            return torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        eps = 1e-8
        temperature = max(float(temperature), eps)
        shared_proto = F.normalize(self.domain_prototypes, dim=-1)
        local_protos = []
        usage_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        for domain_id, (seq, adj) in enumerate(zip(seq_list, adj_list)):
            h = self.gcn(seq[0], adj[0], sparse)
            if h.dim() == 3:
                h = h.squeeze(0)
            if node_indices is not None:
                h = h[node_indices[domain_id]]
            h = F.normalize(h, dim=-1)

            assign_logits = torch.matmul(h, shared_proto.t()) / temperature
            assign_probs = F.softmax(assign_logits, dim=-1)
            with torch.no_grad():
                max_prob = assign_probs.max(dim=1).values.mean().item()
                entropy = (-assign_probs * torch.log(assign_probs + eps)).sum(dim=1).mean().item()
                usage = assign_probs.mean(dim=0)
                usage_min = usage.min().item()
                usage_max = usage.max().item()
                logger.debug(
                    'soft-assign domain=%d nodes=%d max_prob=%.4f entropy=%.4f usage[min=%.4f max=%.4f]',
                    domain_id,
                    assign_probs.shape[0],
                    max_prob,
                    entropy,
                    usage_min,
                    usage_max,
                )
            cluster_mass = assign_probs.sum(dim=0).clamp_min(eps).unsqueeze(-1)
            local_proto = torch.matmul(assign_probs.t(), h) / cluster_mass
            local_proto = F.normalize(local_proto, dim=-1)
            local_protos.append(local_proto)

            usage = assign_probs.mean(dim=0).clamp_min(eps)
            usage_loss = usage_loss + torch.sum(usage * torch.log(usage * self.domain_align_clusters))

        negclip_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        pair_count = 0

        for local_proto in local_protos:
            negclip_loss = negclip_loss + self._negclip_pair_loss(
                local_proto,
                shared_proto,
                torch.cat((shared_proto, *local_protos), dim=0),
                temperature,
            )
            pair_count += 1

        for i in range(len(local_protos)):
            for j in range(i + 1, len(local_protos)):
                hard_bank = torch.cat((shared_proto, local_protos[i], local_protos[j]), dim=0)
                negclip_loss = negclip_loss + self._negclip_pair_loss(local_protos[i], local_protos[j], hard_bank, temperature)
                negclip_loss = negclip_loss + self._negclip_pair_loss(local_protos[j], local_protos[i], hard_bank, temperature)
                pair_count += 2

        if pair_count > 0:
            negclip_loss = negclip_loss / pair_count
        usage_loss = usage_loss / max(len(local_protos), 1)
        return negclip_loss + balance_weight * usage_loss

    def compute_graphcl_node_embeddings(self, domain_id, seq, adj, sparse, node_idx=None):
        """Return two augmented node views for node-level GraphCL InfoNCE."""
        fea_pretext = self.feature_prompt_layers[domain_id]
        str_layers = self.structure_prompt_layers[domain_id]

        if self.ablation_choice == 'None':
            z1 = self.gcn(seq[0], adj[1], sparse)
            z2 = self.gcn(seq[0], adj[2], sparse)
        else:
            fea_z1 = self.gcn(fea_pretext(seq[0]), adj[1], sparse)
            fea_z2 = self.gcn(fea_pretext(seq[0]), adj[2], sparse)
            str_z1 = self.gcn(seq[0], adj[1], sparse, str_layers)
            str_z2 = self.gcn(seq[0], adj[2], sparse, str_layers)
            z1 = self.ablation(fea_z1, str_z1)
            z2 = self.ablation(fea_z2, str_z2)

        if z1.dim() == 3:
            z1 = z1.squeeze(0)
        if z2.dim() == 3:
            z2 = z2.squeeze(0)
        if node_idx is not None:
            z1 = z1[node_idx]
            z2 = z2[node_idx]
        return F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

    def _graphcl_bce_loss(self, logits, target, anchor_weights=None, pos_num=None):
        """BCE loss with optional soft denoising weights on anchor positives."""
        if anchor_weights is None:
            return self.loss(logits, target)
        per_element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        if pos_num is None:
            pos_num = int(logits.shape[1] // 2)
        weights = torch.ones_like(per_element)
        anchor_weights = anchor_weights.to(logits.device).float().clamp_min(1e-6)
        weight_num = min(pos_num, int(anchor_weights.numel()), weights.shape[1])
        if weight_num > 0:
            weights[:, :weight_num] = anchor_weights[:weight_num].view(1, -1)
        return (per_element * weights).sum() / weights.sum().clamp_min(1e-6)

    def graphcl_inbatch_nce_loss(
        self,
        seq_list,
        adj_list,
        sparse,
        node_indices=None,
        sample_weights=None,
        negative_masks=None,
        global_negative_mask=None,
        temperature=0.2,
        domain_ids=None,
    ):
        """Node-level InfoNCE with optional denoised negative masks.

        Positives are the same selected node in two augmented views. By default,
        negatives are all other selected nodes in the current batch. When
        ``negative_masks`` is given, only mask=True off-diagonal pairs are used
        as negatives.
        """
        temperature = max(float(temperature), 1e-8)
        z1_list = []
        z2_list = []
        weight_list = []
        mask_list = []
        if domain_ids is None:
            domain_ids = list(range(len(seq_list)))
        for local_id, (domain_id, seq, adj) in enumerate(zip(domain_ids, seq_list, adj_list)):
            node_idx = None if node_indices is None else node_indices[local_id]
            z1, z2 = self.compute_graphcl_node_embeddings(domain_id, seq, adj, sparse, node_idx=node_idx)
            z1_list.append(z1)
            z2_list.append(z2)
            if sample_weights is not None:
                weight_list.append(sample_weights[local_id].to(z1.device).float())
            if negative_masks is not None:
                mask_list.append(negative_masks[local_id].to(z1.device).bool())

        z1 = torch.cat(z1_list, dim=0)
        z2 = torch.cat(z2_list, dim=0)
        batch_num = z1.shape[0]
        if batch_num <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=z1.device)

        logits_12 = torch.matmul(z1, z2.t()) / temperature
        logits_21 = torch.matmul(z2, z1.t()) / temperature
        if global_negative_mask is not None:
            safe_negative_mask = global_negative_mask.to(z1.device).bool()
            if safe_negative_mask.shape != (batch_num, batch_num):
                raise ValueError(
                    f"global_negative_mask shape mismatch: expected {(batch_num, batch_num)}, got {tuple(safe_negative_mask.shape)}"
                )
            eye = torch.eye(batch_num, dtype=torch.bool, device=z1.device)
            allowed = safe_negative_mask | eye
            logits_12 = logits_12.masked_fill(~allowed, -1e9)
            logits_21 = logits_21.masked_fill(~allowed, -1e9)
        elif negative_masks is not None and mask_list:
            if len(mask_list) == 1:
                safe_negative_mask = mask_list[0]
            else:
                safe_negative_mask = torch.block_diag(*mask_list)
            eye = torch.eye(batch_num, dtype=torch.bool, device=z1.device)
            allowed = safe_negative_mask | eye
            logits_12 = logits_12.masked_fill(~allowed, -1e9)
            logits_21 = logits_21.masked_fill(~allowed, -1e9)
        target = torch.arange(batch_num, device=z1.device)
        loss_12 = F.cross_entropy(logits_12, target, reduction='none')
        loss_21 = F.cross_entropy(logits_21, target, reduction='none')
        per_sample_loss = 0.5 * (loss_12 + loss_21)

        if sample_weights is not None and weight_list:
            weights = torch.cat(weight_list, dim=0).to(per_sample_loss.device).clamp_min(1e-6)
            return (per_sample_loss * weights).sum() / weights.sum()
        return per_sample_loss.mean()

    def graphcl_domain_loss(
        self,
        domain_id,
        seq,
        adj,
        sparse,
        msk,
        samp_bias1,
        samp_bias2,
        lbl,
        node_idx=None,
        neg_idx=None,
        domain_align_lambda=0.0,
        domain_align_temperature=0.2,
        domain_align_balance_weight=0.05,
        graphcl_loss_mode='bce',
        graphcl_nce_lambda=0.0,
        graphcl_nce_temperature=0.2,
        graphcl_sample_weights=None,
        graphcl_negative_mask=None,
        graphcl_anchor_weights=None,
        return_parts=False,
    ):
        """Compute GRAPHCL loss for one domain to allow per-domain backward."""
        logits = next(
            self.compute_prelogits_GRAPHCL(
                [self.feature_prompt_layers[domain_id]],
                [self.structure_prompt_layers[domain_id]],
                [seq],
                [adj],
                sparse,
                msk,
                samp_bias1,
                samp_bias2,
                [node_idx] if node_idx is not None else None,
                [neg_idx] if neg_idx is not None else None,
            )
        )

        target = lbl
        pos_num = logits.shape[1] // 2 if node_idx is None else int(node_idx.numel())
        if target is None or target.shape != logits.shape:
            neg_num = logits.shape[1] - pos_num
            target = torch.cat(
                (
                    torch.ones(logits.shape[0], pos_num, device=logits.device),
                    torch.zeros(logits.shape[0], neg_num, device=logits.device),
                ),
                dim=1,
            )
        graphcl_loss = self._graphcl_bce_loss(
            logits,
            target,
            anchor_weights=graphcl_anchor_weights,
            pos_num=pos_num,
        )
        align_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)

        if domain_align_lambda > 0:
            align_loss = self.compute_domain_shared_prototype_loss(
                seq,
                adj,
                sparse,
                node_idx=node_idx,
                temperature=domain_align_temperature,
                balance_weight=domain_align_balance_weight,
            )
        nce_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        if graphcl_loss_mode in ('nce', 'bce_plus_nce') and graphcl_nce_lambda > 0:
            weights = [graphcl_sample_weights] if graphcl_sample_weights is not None else None
            nce_loss = self.graphcl_inbatch_nce_loss(
                [seq],
                [adj],
                sparse,
                node_indices=[node_idx] if node_idx is not None else None,
                sample_weights=weights,
                negative_masks=[graphcl_negative_mask] if graphcl_negative_mask is not None else None,
                temperature=graphcl_nce_temperature,
                domain_ids=[domain_id],
            )
        if graphcl_loss_mode == 'bce':
            total_loss = graphcl_loss + domain_align_lambda * align_loss
        elif graphcl_loss_mode == 'nce':
            total_loss = graphcl_nce_lambda * nce_loss + domain_align_lambda * align_loss
        elif graphcl_loss_mode == 'bce_plus_nce':
            total_loss = graphcl_loss + graphcl_nce_lambda * nce_loss + domain_align_lambda * align_loss
        else:
            raise ValueError(f"Unsupported graphcl_loss_mode: {graphcl_loss_mode}")
        if return_parts:
            return total_loss, graphcl_loss, align_loss, nce_loss
        return total_loss

    def embed(self, seq, adj, sparse, msk, LP):
        """得到节点 embedding h_1 及图级向量 c（readout）。

        返回 detach 后的张量，通常用于下游评测（不反传梯度）。
        """
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

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
        samples=None,
        graphcl_node_indices=None,
        graphcl_negative_indices=None,
        domain_align_lambda=0.0,
        domain_align_temperature=0.2,
        domain_align_balance_weight=0.05,
        graphcl_loss_mode='bce',
        graphcl_nce_lambda=0.0,
        graphcl_nce_temperature=0.2,
        graphcl_sample_weights=None,
        graphcl_negative_masks=None,
        graphcl_global_negative_mask=None,
        graphcl_anchor_weights=None,
        return_parts=False,
    ):
        """预训练 forward：根据是否提供 samples 来区分 GRAPHCL 与 LP。

        Args:
            seq_list/adj_list: 预训练数据列表（多数据集）
            lbl: GRAPHCL 的对比标签（list，与 seq_list 对齐）
            samples: LP/splitLP 的负采样 tuples
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        graphcl_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        align_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        nce_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        # GRAPHCL 路径：samples==None
        if samples == None:
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
                    graphcl_node_indices,
                    graphcl_negative_indices,
                )
            )
            for i in range(len(logits)):
                target = lbl[i]
                if graphcl_node_indices is None:
                    pos_num = logits[i].shape[1] // 2
                else:
                    pos_num = int(graphcl_node_indices[i].numel())
                if target is None or target.shape != logits[i].shape:
                    neg_num = logits[i].shape[1] - pos_num
                    target = torch.cat(
                        (
                            torch.ones(logits[i].shape[0], pos_num, device=logits[i].device),
                            torch.zeros(logits[i].shape[0], neg_num, device=logits[i].device),
                        ),
                        dim=1,
                    )
                anchor_weights = None if graphcl_anchor_weights is None else graphcl_anchor_weights[i]
                loss = self._graphcl_bce_loss(
                    logits[i],
                    target,
                    anchor_weights=anchor_weights,
                    pos_num=pos_num,
                )
                graphcl_loss += loss

            if domain_align_lambda > 0:
                align_loss = self.compute_domain_prototype_loss(
                    seq_list,
                    adj_list,
                    sparse,
                    node_indices=graphcl_node_indices,
                    temperature=domain_align_temperature,
                    balance_weight=domain_align_balance_weight,
                )
            if graphcl_loss_mode in ('nce', 'bce_plus_nce') and graphcl_nce_lambda > 0:
                nce_loss = self.graphcl_inbatch_nce_loss(
                    seq_list,
                    adj_list,
                    sparse,
                    node_indices=graphcl_node_indices,
                    sample_weights=graphcl_sample_weights,
                    negative_masks=graphcl_negative_masks,
                    global_negative_mask=graphcl_global_negative_mask,
                    temperature=graphcl_nce_temperature,
                )
            if graphcl_loss_mode == 'bce':
                total_loss = graphcl_loss + domain_align_lambda * align_loss
            elif graphcl_loss_mode == 'nce':
                total_loss = graphcl_nce_lambda * nce_loss + domain_align_lambda * align_loss
            elif graphcl_loss_mode == 'bce_plus_nce':
                total_loss = graphcl_loss + graphcl_nce_lambda * nce_loss + domain_align_lambda * align_loss
            else:
                raise ValueError(f"Unsupported graphcl_loss_mode: {graphcl_loss_mode}")

        # LP / splitLP 路径
        else:
            logits = list(
                self.compute_prelogits_LP(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    sparse,
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

        if return_parts:
            return total_loss, graphcl_loss, align_loss, nce_loss
        return total_loss



def _to_numpy_features(feature):
    """Convert cached/preprocessed node features to a CPU numpy matrix."""
    if torch.is_tensor(feature):
        return feature.detach().cpu().numpy()
    return np.asarray(feature)


def cluster_select_high_quality_nodes(
    feature,
    keep_ratio=1.0,
    n_clusters=10,
    min_nodes=1,
    random_state=0,
    keep_num=None,
    kmeans_batch_size=10000,
    return_details=False,
):
    """Select high-quality nodes by cluster centrality inside one domain.

    Nodes nearest to their KMeans centroid are treated as high-confidence,
    representative samples. The quota is assigned proportionally across
    clusters with at least one node per non-empty selected cluster, so dense
    clusters do not consume the entire domain quota.
    """
    node_num = feature.shape[0]
    if keep_num is None:
        keep_num = int(math.ceil(node_num * keep_ratio))
    keep_num = int(keep_num)
    keep_num = max(min_nodes, keep_num)
    keep_num = min(node_num, keep_num)
    if keep_num >= node_num or node_num <= 1:
        selected = np.arange(node_num, dtype=np.int64)
        if return_details:
            return selected, {
                'labels': np.zeros(node_num, dtype=np.int64),
                'centers': None,
                'cluster_num': 1 if node_num else 0,
                'keep_num': len(selected),
                'node_num': node_num,
                'method': 'all_nodes',
            }
        return selected

    feat_np = _to_numpy_features(feature)
    cluster_num = max(1, min(n_clusters, keep_num, node_num))
    if kmeans_batch_size and node_num > kmeans_batch_size:
        method = 'MiniBatchKMeans'
        kmeans = MiniBatchKMeans(
            n_clusters=cluster_num,
            random_state=random_state,
            batch_size=kmeans_batch_size,
            n_init=3,
        )
    else:
        method = 'KMeans'
        kmeans = KMeans(n_clusters=cluster_num, random_state=random_state, n_init=10)
    logger.debug(
        'cluster_select_high_quality_nodes: nodes=%s keep_num=%s clusters=%s method=%s',
        node_num,
        keep_num,
        cluster_num,
        method,
    )
    labels = kmeans.fit_predict(feat_np)
    centers = kmeans.cluster_centers_

    selected = []
    leftovers = []
    for cluster_id in range(cluster_num):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        distances = np.linalg.norm(feat_np[members] - centers[cluster_id], axis=1)
        ordered = members[np.argsort(distances)]
        quota = max(1, int(round(keep_num * len(members) / node_num)))
        quota = min(quota, len(ordered), keep_num - len(selected))
        selected.extend(ordered[:quota].tolist())
        leftovers.extend(ordered[quota:].tolist())

    if len(selected) < keep_num:
        selected.extend(leftovers[: keep_num - len(selected)])
    selected = np.asarray(selected[:keep_num], dtype=np.int64)
    if return_details:
        return selected, {
            'labels': labels,
            'centers': centers,
            'cluster_num': cluster_num,
            'keep_num': keep_num,
            'node_num': node_num,
            'method': method,
        }
    return selected


def visualize_cluster_filtering(feature, selected_indices, labels=None, output_path=None, title='cluster_filtering', random_state=0):
    """Save a before/after cluster-filtering visualization as a PNG image."""
    if output_path is None:
        return None

    feat_np = _to_numpy_features(feature)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import matplotlib

        matplotlib.use('Agg')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        logger.warning('visualize_cluster_filtering: matplotlib is unavailable: %s', exc)
        return None

    if feat_np.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=random_state).fit_transform(feat_np)
    else:
        coords = np.column_stack([feat_np[:, 0], np.zeros(feat_np.shape[0])])

    mask = np.zeros(feat_np.shape[0], dtype=bool)
    mask[selected_indices] = True

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if labels is None:
        axes[0].scatter(coords[:, 0], coords[:, 1], s=5, c='lightgray', alpha=0.8)
    else:
        axes[0].scatter(coords[:, 0], coords[:, 1], s=5, c=labels, cmap='tab20', alpha=0.8)
        unique_labels = np.unique(labels)
        if unique_labels.size > 0:
            max_label = int(unique_labels.max())
            cmap = plt.cm.get_cmap('tab20', max_label + 1)
            handles = [
                Line2D([0], [0], marker='o', linestyle='', markersize=4, color=cmap(int(label)),
                       label=f'cluster {int(label)}')
                for label in unique_labels
            ]
            axes[0].legend(handles=handles, title='cluster id', loc='best', fontsize=7, title_fontsize=8, framealpha=0.85)
    axes[0].set_title(f'{title} - before ({feat_np.shape[0]} nodes)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(coords[~mask, 0], coords[~mask, 1], s=4, c='lightgray', alpha=0.25, label='filtered out')
    axes[1].scatter(coords[mask, 0], coords[mask, 1], s=8, c='crimson', alpha=0.9, label='selected')
    axes[1].set_title(f'{title} - after ({mask.sum()} selected)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(loc='best')

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info('cluster visualization saved: %s', output_path)
    return output_path


def build_balanced_domain_node_indices(
    features,
    keep_ratio=1.0,
    n_clusters=10,
    min_nodes=1,
    random_state=0,
    balance_mode='fixed',
    nodes_per_domain=0,
    kmeans_batch_size=10000,
    visualize=False,
    visualization_dir=None,
    domain_names=None,
):
    """Build per-domain cluster-selected node pools with cross-domain balance.

    The cross-domain problem is caused by comparing all domains with one global
    threshold: large/easy domains can contribute many high-score nodes while
    hard domains contribute very few. We therefore cluster and rank nodes inside
    each domain first, then assign an explicit domain quota.

    Args:
        balance_mode: ``'fixed'`` gives every domain the same quota; ``'ratio'``
            keeps the same per-domain ratio but may still produce different
            absolute counts for differently sized domains.
        nodes_per_domain: explicit quota used in ``'fixed'`` mode. If <= 0, the
            quota is the smallest ratio-based quota across all domains, so no
            domain is asked to provide more selected nodes than it has.
    """
    if keep_ratio >= 1.0 and not (balance_mode == 'fixed' and nodes_per_domain and nodes_per_domain > 0):
        logger.debug('build_balanced_domain_node_indices: keep_ratio>=1.0 and no explicit quota; use all nodes')
        return [np.arange(feature.shape[0], dtype=np.int64) for feature in features]

    ratio_quotas = [
        min(feature.shape[0], max(min_nodes, int(math.ceil(feature.shape[0] * keep_ratio))))
        for feature in features
    ]
    if balance_mode == 'fixed':
        if nodes_per_domain and nodes_per_domain > 0:
            domain_quotas = [min(feature.shape[0], max(min_nodes, nodes_per_domain)) for feature in features]
        else:
            shared_quota = min(ratio_quotas)
            domain_quotas = [min(feature.shape[0], shared_quota) for feature in features]
    elif balance_mode == 'ratio':
        domain_quotas = ratio_quotas
    else:
        raise ValueError(f"Unsupported cluster balance mode: {balance_mode}")

    logger.debug(
        'build_balanced_domain_node_indices: balance_mode=%s keep_ratio=%s ratio_quotas=%s domain_quotas=%s',
        balance_mode,
        keep_ratio,
        ratio_quotas,
        domain_quotas,
    )
    domain_names = domain_names or [f'domain_{i}' for i in range(len(features))]
    selected_pools = []
    for domain_id, feature in enumerate(features):
        selected, details = cluster_select_high_quality_nodes(
            feature,
            keep_ratio=keep_ratio,
            n_clusters=n_clusters,
            min_nodes=min_nodes,
            random_state=random_state + domain_id,
            keep_num=domain_quotas[domain_id],
            kmeans_batch_size=kmeans_batch_size,
            return_details=True,
        )
        selected_pools.append(selected)
        logger.debug(
            'domain %s cluster filtering: nodes=%s selected=%s clusters=%s method=%s',
            domain_names[domain_id],
            details['node_num'],
            len(selected),
            details['cluster_num'],
            details['method'],
        )
        if visualize and visualization_dir:
            safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(domain_names[domain_id]))
            output_path = os.path.join(visualization_dir, f'{domain_id:02d}_{safe_name}_cluster_filtering.png')
            visualize_cluster_filtering(
                feature,
                selected,
                labels=details.get('labels'),
                output_path=output_path,
                title=str(domain_names[domain_id]),
                random_state=random_state + domain_id,
            )
    return selected_pools


def sample_graphcl_minibatches(node_pools, batch_size, negative_ratio=1, device=None):
    """Sample anchor mini-batches and negatives from other selected nodes."""
    batch_indices = []
    negative_indices = []
    for pool in node_pools:
        pool = np.asarray(pool, dtype=np.int64)
        if len(pool) == 0:
            pool = np.asarray([0], dtype=np.int64)
        anchor_num = len(pool) if batch_size <= 0 else min(batch_size, len(pool))
        anchors = np.random.choice(pool, size=anchor_num, replace=False)
        candidates = np.setdiff1d(pool, anchors, assume_unique=False)
        if len(candidates) == 0:
            candidates = pool
        neg_num = max(anchor_num, anchor_num * max(1, negative_ratio))
        negatives = np.random.choice(candidates, size=neg_num, replace=len(candidates) < neg_num)
        batch_indices.append(torch.as_tensor(anchors, dtype=torch.long, device=device))
        negative_indices.append(torch.as_tensor(negatives, dtype=torch.long, device=device))
    return batch_indices, negative_indices


def compute_cluster_quality_weights(
    features,
    n_clusters=10,
    random_state=0,
    kmeans_batch_size=10000,
):
    """Compute per-node quality weights from within-domain cluster centrality.

    The score is domain-local, so domains with larger feature scale or easier
    clusters do not automatically dominate. Values are normalized to mean 1 in
    each domain; selected central nodes get weights above 1, boundary/noisy
    nodes below 1. These weights are used softly in InfoNCE instead of making
    clustering the only gate into training.
    """
    all_weights = []
    for domain_id, feature in enumerate(features):
        feat_np = _to_numpy_features(feature)
        node_num = feat_np.shape[0]
        if node_num <= 1:
            all_weights.append(np.ones(node_num, dtype=np.float32))
            continue
        cluster_num = max(1, min(n_clusters, node_num))
        if kmeans_batch_size and node_num > kmeans_batch_size:
            kmeans = MiniBatchKMeans(
                n_clusters=cluster_num,
                random_state=random_state + domain_id,
                batch_size=kmeans_batch_size,
                n_init=3,
            )
        else:
            kmeans = KMeans(n_clusters=cluster_num, random_state=random_state + domain_id, n_init=10)
        labels = kmeans.fit_predict(feat_np)
        centers = kmeans.cluster_centers_
        weights = np.ones(node_num, dtype=np.float32)
        for cluster_id in range(cluster_num):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                continue
            distances = np.linalg.norm(feat_np[members] - centers[cluster_id], axis=1)
            scale = np.median(distances) + 1e-8
            weights[members] = np.exp(-distances / scale).astype(np.float32)
        weights = weights / max(float(weights.mean()), 1e-8)
        all_weights.append(weights.astype(np.float32))
    return all_weights


def compute_cluster_denoising_metadata(
    features,
    n_clusters=10,
    random_state=0,
    kmeans_batch_size=10000,
    weight_alpha=1.0,
    weight_min=0.25,
    weight_max=2.0,
):
    """Build static cluster-guided denoising signals for each domain.

    The metadata intentionally uses weak signals only:
    - ``quality_weights`` are domain-normalized soft reliability weights.
    - ``labels`` are used to avoid treating nearby/same-cluster samples as
      strong negatives.
    - ``feature_norm`` provides a cheap, fixed similarity space for masking.

    We avoid using these clusters as hard pseudo labels or strong alignment
    targets, because in cross-domain graphs clusters may mix structure,
    attributes, and domain-specific prompt effects.
    """
    metadata = []
    for domain_id, feature in enumerate(features):
        feat_np = _to_numpy_features(feature).astype(np.float32)
        node_num = feat_np.shape[0]
        if node_num <= 1:
            metadata.append(
                {
                    'labels': np.zeros(node_num, dtype=np.int64),
                    'quality_weights': np.ones(node_num, dtype=np.float32),
                    'feature_norm': feat_np,
                    'cluster_num': 1 if node_num else 0,
                    'method': 'single_node',
                    'quality_mean': 1.0,
                    'quality_std': 0.0,
                    'margin_mean': 0.0,
                    'margin_std': 0.0,
                }
            )
            continue

        cluster_num = max(2, min(n_clusters, node_num))
        if kmeans_batch_size and node_num > kmeans_batch_size:
            method = 'MiniBatchKMeans'
            kmeans = MiniBatchKMeans(
                n_clusters=cluster_num,
                random_state=random_state + domain_id,
                batch_size=kmeans_batch_size,
                n_init=3,
            )
        else:
            method = 'KMeans'
            kmeans = KMeans(n_clusters=cluster_num, random_state=random_state + domain_id, n_init=10)
        labels = kmeans.fit_predict(feat_np)
        centers = kmeans.cluster_centers_.astype(np.float32)

        feat_norm = feat_np / np.maximum(np.linalg.norm(feat_np, axis=1, keepdims=True), 1e-8)
        center_norm = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        sims = np.matmul(feat_norm, center_norm.T)
        assigned_sim = sims[np.arange(node_num), labels]
        if cluster_num > 1:
            sims_without_assigned = sims.copy()
            sims_without_assigned[np.arange(node_num), labels] = -np.inf
            second_sim = np.max(sims_without_assigned, axis=1)
            margin = assigned_sim - second_sim
        else:
            margin = assigned_sim

        median = float(np.median(margin))
        mad = float(np.median(np.abs(margin - median))) + 1e-6
        raw_quality = 1.0 / (1.0 + np.exp(-(margin - median) / mad))
        quality = 1.0 + float(weight_alpha) * (raw_quality - float(raw_quality.mean()))
        quality = np.clip(quality, weight_min, weight_max)
        quality = quality / max(float(quality.mean()), 1e-8)

        metadata.append(
            {
                'labels': labels.astype(np.int64),
                'quality_weights': quality.astype(np.float32),
                'feature_norm': feat_norm.astype(np.float32),
                'cluster_num': cluster_num,
                'method': method,
                'quality_mean': float(quality.mean()),
                'quality_std': float(quality.std()),
                'quality_min': float(quality.min()),
                'quality_max': float(quality.max()),
                'margin_mean': float(margin.mean()),
                'margin_std': float(margin.std()),
                'margin_min': float(margin.min()),
                'margin_max': float(margin.max()),
            }
        )
        logger.debug(
            'cluster denoising metadata domain=%d nodes=%d clusters=%d method=%s '
            'quality[min=%.4f mean=%.4f max=%.4f std=%.4f] margin[mean=%.4f std=%.4f]',
            domain_id,
            node_num,
            cluster_num,
            method,
            metadata[-1]['quality_min'],
            metadata[-1]['quality_mean'],
            metadata[-1]['quality_max'],
            metadata[-1]['quality_std'],
            metadata[-1]['margin_mean'],
            metadata[-1]['margin_std'],
        )
    return metadata


def visualize_denoising_metadata(feature, metadata, output_path=None, title='cluster_denoising', random_state=0):
    """Save a compact visualization for cluster-guided denoising signals."""
    if output_path is None:
        return None
    feat_np = _to_numpy_features(feature)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import matplotlib

        matplotlib.use('Agg')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning('visualize_denoising_metadata: matplotlib is unavailable: %s', exc)
        return None

    if feat_np.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=random_state).fit_transform(feat_np)
    else:
        coords = np.column_stack([feat_np[:, 0], np.zeros(feat_np.shape[0])])

    labels = metadata['labels']
    weights = metadata['quality_weights']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(coords[:, 0], coords[:, 1], s=5, c=labels, cmap='tab20', alpha=0.85)
    axes[0].set_title(f'{title} clusters')
    axes[1].scatter(coords[:, 0], coords[:, 1], s=5, c=weights, cmap='viridis', alpha=0.85)
    axes[1].set_title('anchor reliability')
    axes[2].hist(weights, bins=30, color='steelblue', alpha=0.9)
    axes[2].set_title('reliability histogram')
    axes[2].set_xlabel('weight')
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info('denoising visualization saved: %s', output_path)
    return output_path


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
