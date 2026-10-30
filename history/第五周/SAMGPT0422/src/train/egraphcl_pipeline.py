import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def node_contrastive_loss(h1, h2, temperature, node_proj):
    """节点级 InfoNCE 对比损失。

    Args:
        h1: 视图1节点 embedding，形状 [N, F]（squeeze 后）
        h2: 视图2节点 embedding，形状 [N, F]
        temperature: 温度参数
        node_proj: 节点级投影头（两层 MLP）

    Returns:
        loss: 标量损失（节点数 < 4 时返回 0）
    """
    # 去掉可能的 batch 维
    if h1.dim() == 3:
        h1 = h1.squeeze(0)
    if h2.dim() == 3:
        h2 = h2.squeeze(0)

    n = h1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=h1.device, dtype=h1.dtype)

    z1 = F.normalize(node_proj(h1), p=2, dim=1)  # [N, D]
    z2 = F.normalize(node_proj(h2), p=2, dim=1)  # [N, D]

    # 拼接两个视图：[2N, D]
    z = torch.cat([z1, z2], dim=0)
    # 相似度矩阵 [2N, 2N]
    sim = torch.matmul(z, z.t()) / float(temperature)

    # 正样本对：i 与 i+N（以及 i+N 与 i）
    labels = torch.arange(n, device=h1.device)
    labels = torch.cat([labels + n, labels], dim=0)  # [2N]

    # 去掉自身相似度（对角线设为 -inf）
    mask = torch.eye(2 * n, device=h1.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss

def _normalize_np_rows(x):
    """按行归一化 numpy 矩阵，避免 0 向量除零。"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return x / norm


def _compute_sample_pair_similarity(sample_features):
    """由样本表示向量计算样本对余弦相似度。"""
    if sample_features is None:
        return None

    z = np.asarray(sample_features, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] <= 1:
        return None

    z = _normalize_np_rows(z)
    return z @ z.T


def build_pair_soft_mask(
    sample_features,
    tau,
    result_dir,
):
    """构建 EGRAPHCL 的样本对 soft mask（纯样本级余弦相似度，不做按簇统计）。"""

    tau_sample = float(tau)
    pair_sim_sample = _compute_sample_pair_similarity(sample_features)
    if pair_sim_sample is None:
        print('[egraphcl] sample-pair mask summary: sample-feature unavailable, fallback to default pair mask')
        return None

    # 基于样本级余弦相似度构造 mask
    # 相似度 >= tau 的样本对被屏蔽
    soft_mask_np = (pair_sim_sample >= tau_sample)

    # 去掉 self-self
    np.fill_diagonal(soft_mask_np, False)

    # 转成 torch tensor，保持与原先下游逻辑兼容
    egraphcl_pair_soft_mask = torch.from_numpy(soft_mask_np.astype(np.float32))

    n = int(soft_mask_np.shape[0])
    off_diag_total = int(n * (n - 1))
    masked_pairs = int(soft_mask_np.sum())
    compared_pairs = int(off_diag_total - masked_pairs)

    print(
        '[egraphcl] sample-pair mask summary: '
        f'tau_sample={tau_sample:.6f}, '
        f'samples={n}, '
        f'masked_soft_pairs={masked_pairs}, '
        f'compared_negative_pairs={compared_pairs}, '
        f'masked_ratio={masked_pairs / max(1, off_diag_total):.6f}'
    )

    # 每个样本被屏蔽的 pair 数量
    per_sample_masked = soft_mask_np.sum(axis=1).astype(np.int64)
    per_sample_total = max(1, n - 1)
    per_sample_ratio = per_sample_masked.astype(np.float64) / float(per_sample_total)

    # ===== 图1：按样本 ID 的 mask 比例曲线 =====
    try:
        fig = plt.figure(figsize=(11, 4))
        seq_ids = np.arange(1, n + 1, dtype=np.int64)
        plt.plot(seq_ids, per_sample_ratio, linewidth=0.8)
        plt.xlabel('id (1..N)')
        plt.ylabel('masked_ratio')
        plt.title('EGRAPHCL Soft-Mask Ratio by Sequential ID')
        plt.grid(alpha=0.25)
        fig.tight_layout()

        by_id_png = os.path.join(result_dir, 'egraphcl_soft_mask_by_sample_id.png')
        fig.savefig(by_id_png, dpi=180)
        plt.close(fig)

        print(f'[egraphcl] saved visualization: {by_id_png}')
    except Exception as e:
        print(f'[egraphcl] warn: failed to save visualization: {e}')

    return egraphcl_pair_soft_mask


def report_fallback_mask_summary(masked_pairs, total_pairs):
    """当使用 node-level mask 时，输出统计信息。"""
    if total_pairs > 0:
        kept_pairs = total_pairs - masked_pairs
        print(
            '[egraphcl] fallback node-level masking summary: '
            f'masked={masked_pairs}, '
            f'kept_negative={kept_pairs}, '
            f'total_negative={total_pairs}, '
            f'masked_ratio={masked_pairs / total_pairs:.6f}'
        )
    else:
        print('[egraphcl] fallback node-level masking summary: no negative pairs found')


def _resolve_route_id(sample_i, sample_dataset_ids, sample_prototype_ids, prompt_routing_mode, prompt_id_split_base):
    """Build unified route id with disjoint id ranges for dataset/prototype routing."""
    route_mode = str(prompt_routing_mode).lower()
    split_base = int(prompt_id_split_base)
    if route_mode == 'prototype':
        raw_id = sample_i if sample_prototype_ids is None else int(sample_prototype_ids[sample_i])
        return split_base + int(raw_id), route_mode
    raw_id = sample_i if sample_dataset_ids is None else int(sample_dataset_ids[sample_i])
    return int(raw_id), route_mode


def _resolve_prompt_index(route_id, route_mode, prompt_bank_size, prompt_id_split_base):
    """Map unified route id to bank-local index according to the active routing mode."""
    split_base = int(prompt_id_split_base)
    if route_mode == 'prototype':
        prompt_idx = int(route_id) - split_base
    else:
        prompt_idx = int(route_id)

    if prompt_idx < 0 or prompt_idx >= int(prompt_bank_size):
        raise IndexError(
            f"route_id={route_id} -> prompt_idx={prompt_idx} out of range for prompt layers "
            f"(size={prompt_bank_size}), mode={route_mode}, split_base={split_base}"
        )
    return prompt_idx


def _normalize_proto_weights(weights, prompt_bank_size, device):
    """Normalize prototype weights to a valid probability vector on target device."""
    if weights is None:
        return None

    if torch.is_tensor(weights):
        w = weights.to(device=device, dtype=torch.float32)
    else:
        w = torch.as_tensor(weights, dtype=torch.float32, device=device)

    if w.ndim != 1:
        w = w.view(-1)

    if w.shape[0] < int(prompt_bank_size):
        pad = torch.zeros(int(prompt_bank_size) - int(w.shape[0]), device=device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=0)
    elif w.shape[0] > int(prompt_bank_size):
        w = w[:int(prompt_bank_size)]

    w = torch.clamp(w, min=0.0)
    denom = torch.sum(w)
    if float(denom.item()) <= 0.0:
        return None
    return w / denom


def _build_proto_weight_matrix(sample_prototype_weights, prompt_bank_size, device):
    """Convert per-sample prototype weights into a normalized [B, K] tensor."""
    if sample_prototype_weights is None:
        return None

    rows = []
    for w in sample_prototype_weights:
        w_norm = _normalize_proto_weights(w, prompt_bank_size, device)
        if w_norm is None:
            return None
        rows.append(w_norm)

    if len(rows) == 0:
        return None
    return torch.stack(rows, dim=0)


def _prototype_regularization(z1, z2, q_prior, tau, w_consistency, w_balance):
    """Prototype-guided regularization from two augmented views.

    - consistency: symmetric KL between view-wise prototype posteriors
    - balance: avoid prototype collapse by maximizing batch assignment entropy
    """
    zero = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    if q_prior is None:
        return zero, {
            'proto_consistency_loss': 0.0,
            'proto_balance_loss': 0.0,
            'proto_entropy': 0.0,
        }

    if float(w_consistency) <= 0.0 and float(w_balance) <= 0.0:
        return zero, {
            'proto_consistency_loss': 0.0,
            'proto_balance_loss': 0.0,
            'proto_entropy': 0.0,
        }

    z1n = F.normalize(z1, p=2, dim=1)
    z2n = F.normalize(z2, p=2, dim=1)
    z_mean = F.normalize((z1n + z2n) / 2.0, p=2, dim=1)

    denom = torch.sum(q_prior, dim=0, keepdim=True).t() + 1e-6
    anchors = torch.matmul(q_prior.t(), z_mean) / denom
    anchors = F.normalize(anchors, p=2, dim=1)

    tau_safe = max(1e-6, float(tau))
    logits1 = torch.matmul(z1n, anchors.t()) / tau_safe
    logits2 = torch.matmul(z2n, anchors.t()) / tau_safe
    q1 = F.softmax(logits1, dim=1)
    q2 = F.softmax(logits2, dim=1)

    log_q1 = torch.log(q1 + 1e-12)
    log_q2 = torch.log(q2 + 1e-12)

    loss_pc = 0.5 * (
        F.kl_div(log_q1, q2, reduction='batchmean')
        + F.kl_div(log_q2, q1, reduction='batchmean')
    )

    q_mix = 0.5 * (q1 + q2)
    q_bar = torch.mean(q_mix, dim=0)
    loss_bal = torch.sum(q_bar * torch.log(q_bar + 1e-12))
    entropy = -torch.sum(q_bar * torch.log(q_bar + 1e-12))

    total = float(w_consistency) * loss_pc + float(w_balance) * loss_bal
    stats = {
        'proto_consistency_loss': float(loss_pc.detach().item()),
        'proto_balance_loss': float(loss_bal.detach().item()),
        'proto_entropy': float(entropy.detach().item()),
    }
    return total, stats


class _SoftTextPrompt(nn.Module):
    """A lightweight prompt module with externally provided blended weight tensor."""

    def __init__(self, blended_weight, prompt_type='mul'):
        super().__init__()
        self._blended_weight = blended_weight
        self.prompttype = prompt_type

    @property
    def weight(self):
        return self._blended_weight

    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            if graph_embedding.ndim >= 2:
                weight = self.weight.repeat(graph_embedding.shape[0], 1)
            else:
                weight = self.weight.view(-1)
            return graph_embedding + weight
        return self.weight * graph_embedding


def _blend_prompt_weight(prompt_layers, coeff):
    """Blend prompt weights using coeff over prompt layers."""
    bank_size = len(prompt_layers)
    if coeff is None:
        return None
    if coeff.shape[0] != bank_size:
        return None
    blended = torch.zeros_like(prompt_layers[0].weight)
    for j in range(bank_size):
        blended = blended + coeff[j] * prompt_layers[j].weight
    return blended


def _apply_soft_feature_prompt(seq_in, feature_prompt_layers, coeff):
    """Apply blended feature prompt directly on input sequence."""
    blended = _blend_prompt_weight(feature_prompt_layers, coeff)
    if blended is None:
        return None
    prompt_type = getattr(feature_prompt_layers[0], 'prompttype', 'mul')
    prompt = _SoftTextPrompt(blended, prompt_type=prompt_type)
    return prompt(seq_in)


def _build_soft_structure_prompts(structure_prompt_layers, coeff):
    """Build per-layer blended structure prompts."""
    bank_size = len(structure_prompt_layers)
    if coeff is None or coeff.shape[0] != bank_size:
        return None
    num_layers = len(structure_prompt_layers[0])
    out = []
    for li in range(num_layers):
        layer_bank = [structure_prompt_layers[j][li] for j in range(bank_size)]
        blended = _blend_prompt_weight(layer_bank, coeff)
        if blended is None:
            return None
        prompt_type = getattr(layer_bank[0], 'prompttype', 'mul')
        out.append(_SoftTextPrompt(blended, prompt_type=prompt_type))
    return out


def compute_graph_pair_vectors(
    gcn,
    read,
    egraphcl_proj,
    feature_prompt_layers,
    structure_prompt_layers,
    seq_list,
    adj_list,
    sparse=False,
    sample_dataset_ids=None,
    sample_prototype_ids=None,
    sample_prototype_weights=None,
    prompt_routing_mode='dataset',
    prompt_id_split_base=10000,
    use_feature_prompt=True,
):
    """Build sample-level graph embeddings for two augmented graph views."""
    if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
        sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()
    if sample_prototype_ids is not None and torch.is_tensor(sample_prototype_ids):
        sample_prototype_ids = sample_prototype_ids.detach().cpu().tolist()

    z1_list = []
    z2_list = []
    for sample_i, (seq, adj) in enumerate(zip(seq_list, adj_list)):
        proto_coeff = None
        if str(prompt_routing_mode).lower() == 'prototype' and sample_prototype_weights is not None:
            proto_coeff = _normalize_proto_weights(
                sample_prototype_weights[sample_i],
                len(feature_prompt_layers),
                seq[0].device,
            )

        route_id, route_mode = _resolve_route_id(
            sample_i,
            sample_dataset_ids,
            sample_prototype_ids,
            prompt_routing_mode,
            prompt_id_split_base,
        )
        prompt_idx = _resolve_prompt_index(route_id, route_mode, len(feature_prompt_layers), prompt_id_split_base)

        fea_pretext = feature_prompt_layers[prompt_idx]
        str_layers = structure_prompt_layers[prompt_idx]

        if use_feature_prompt:
            if proto_coeff is not None:
                seq_soft = _apply_soft_feature_prompt(seq[0], feature_prompt_layers, proto_coeff)
                seq_in = seq_soft if seq_soft is not None else fea_pretext(seq[0])
            else:
                seq_in = fea_pretext(seq[0])
            prompt_layers = None
        else:
            seq_in = seq[0]
            if proto_coeff is not None:
                prompt_soft = _build_soft_structure_prompts(structure_prompt_layers, proto_coeff)
                prompt_layers = prompt_soft if prompt_soft is not None else str_layers
            else:
                prompt_layers = str_layers

        h1 = gcn(seq_in, adj[1], sparse, prompt_layers=prompt_layers)
        h2 = gcn(seq_in, adj[2], sparse, prompt_layers=prompt_layers)
        c1 = read(h1, None).view(-1)
        c2 = read(h2, None).view(-1)
        z1_list.append(c1)
        z2_list.append(c2)

    z1 = torch.stack(z1_list, dim=0)
    z2 = torch.stack(z2_list, dim=0)
    z1 = egraphcl_proj(z1)
    z2 = egraphcl_proj(z2)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    return z1, z2


def _to_dense_adj(adj_t):
    if adj_t.is_sparse:
        return adj_t.to_dense()
    return adj_t


def _build_block_diag_dense(adj_list):
    if len(adj_list) == 1:
        return _to_dense_adj(adj_list[0]), [int(adj_list[0].shape[0])]

    dense = [_to_dense_adj(a) for a in adj_list]
    sizes = [int(a.shape[0]) for a in dense]
    return torch.block_diag(*dense), sizes


def _segment_means(h, sizes):
    # h: [1, total_nodes, hidden]
    h = h.squeeze(0)
    outs = []
    start = 0
    for n in sizes:
        end = start + int(n)
        outs.append(h[start:end].mean(dim=0))
        start = end
    return torch.stack(outs, dim=0)


def _iter_chunks(indices, chunk_size):
    if chunk_size <= 0 or chunk_size >= len(indices):
        yield indices
        return
    for start in range(0, len(indices), chunk_size):
        yield indices[start:start + chunk_size]


class _ReadoutProxy:
    """Minimal readout adapter for per-sample fallback path."""

    def __call__(self, h, _msk=None):
        if h.ndim == 3:
            return torch.mean(h, dim=1)
        if h.ndim == 2:
            return torch.mean(h, dim=0, keepdim=True)
        return h


def _build_soft_prompt_layers_for_coeff(structure_prompt_layers, coeff, route_idx):
    """Return per-layer prompt modules for a coefficient vector.

    If coeff is invalid, fallback to hard-routed prompt layers.
    """
    prompt_soft = _build_soft_structure_prompts(structure_prompt_layers, coeff)
    if prompt_soft is not None:
        return prompt_soft
    return structure_prompt_layers[route_idx]


def compute_graph_pair_vectors_batched(
    gcn,
    egraphcl_proj,
    feature_prompt_layers,
    structure_prompt_layers,
    seq_list,
    adj_list,
    sparse=False,
    sample_dataset_ids=None,
    sample_prototype_ids=None,
    sample_prototype_weights=None,
    prompt_routing_mode='dataset',
    prompt_id_split_base=10000,
    batched_chunk_graphs=0,
    use_feature_prompt=True,
):
    """Batched graph-vector extraction.

    - feature branch: full batch via block-diagonal adjacency
    - structure branch: grouped by route_id to preserve per-route prompt layers
    """
    if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
        sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()
    if sample_prototype_ids is not None and torch.is_tensor(sample_prototype_ids):
        sample_prototype_ids = sample_prototype_ids.detach().cpu().tolist()

    prototype_soft_mode = (
        str(prompt_routing_mode).lower() == 'prototype' and sample_prototype_weights is not None
    )

    route_ids = []
    for sample_i in range(len(seq_list)):
        route_id, route_mode = _resolve_route_id(
            sample_i,
            sample_dataset_ids,
            sample_prototype_ids,
            prompt_routing_mode,
            prompt_id_split_base,
        )
        prompt_idx = _resolve_prompt_index(route_id, route_mode, len(feature_prompt_layers), prompt_id_split_base)
        route_ids.append(prompt_idx)

    device = seq_list[0].device
    bsz = len(seq_list)
    chunk_graphs = int(batched_chunk_graphs)
    z1_out = torch.zeros((bsz, egraphcl_proj[0].in_features), device=device)
    z2_out = torch.zeros((bsz, egraphcl_proj[0].in_features), device=device)

    if use_feature_prompt:
        for chunk_indices in _iter_chunks(list(range(bsz)), chunk_graphs):
            seq_inputs = []
            for i in chunk_indices:
                if prototype_soft_mode:
                    coeff = _normalize_proto_weights(
                        sample_prototype_weights[i],
                        len(feature_prompt_layers),
                        seq_list[i][0].device,
                    )
                    if coeff is not None:
                        seq_soft = _apply_soft_feature_prompt(seq_list[i][0], feature_prompt_layers, coeff)
                        if seq_soft is not None:
                            seq_inputs.append(seq_soft)
                            continue
                seq_inputs.append(feature_prompt_layers[route_ids[i]](seq_list[i][0]))

            adj1_list = [adj_list[i][1] for i in chunk_indices]
            adj2_list = [adj_list[i][2] for i in chunk_indices]

            cat_x = torch.cat(seq_inputs, dim=0)
            adj1_blk, sizes = _build_block_diag_dense(adj1_list)
            adj2_blk, _ = _build_block_diag_dense(adj2_list)

            h1 = gcn(cat_x, adj1_blk, sparse=False, prompt_layers=None)
            h2 = gcn(cat_x, adj2_blk, sparse=False, prompt_layers=None)
            z1_c = _segment_means(h1, sizes)
            z2_c = _segment_means(h2, sizes)

            for local_i, sample_i in enumerate(chunk_indices):
                z1_out[sample_i] = z1_c[local_i]
                z2_out[sample_i] = z2_c[local_i]
    else:
        groups = defaultdict(list)

        if prototype_soft_mode:
            # Group by exact normalized prototype coefficients so each group can still run batched.
            for i in range(bsz):
                coeff = _normalize_proto_weights(
                    sample_prototype_weights[i],
                    len(feature_prompt_layers),
                    seq_list[i][0].device,
                )
                if coeff is None:
                    key = ('hard', int(route_ids[i]))
                else:
                    key = ('soft', tuple(torch.round(coeff.detach().cpu(), decimals=6).tolist()))
                groups[key].append(i)
        else:
            for i, rid in enumerate(route_ids):
                groups[('hard', int(rid))].append(i)

        for key, indices in groups.items():
            for chunk_indices in _iter_chunks(indices, chunk_graphs):
                seq_inputs = [seq_list[i][0] for i in chunk_indices]
                adj1_list = [adj_list[i][1] for i in chunk_indices]
                adj2_list = [adj_list[i][2] for i in chunk_indices]

                cat_x = torch.cat(seq_inputs, dim=0)
                adj1_blk, sizes = _build_block_diag_dense(adj1_list)
                adj2_blk, _ = _build_block_diag_dense(adj2_list)

                if key[0] == 'soft':
                    sample_i0 = chunk_indices[0]
                    coeff0 = _normalize_proto_weights(
                        sample_prototype_weights[sample_i0],
                        len(feature_prompt_layers),
                        seq_list[sample_i0][0].device,
                    )
                    prompt_layers = _build_soft_prompt_layers_for_coeff(
                        structure_prompt_layers,
                        coeff0,
                        route_ids[sample_i0],
                    )
                else:
                    prompt_layers = structure_prompt_layers[int(key[1])]

                h1 = gcn(cat_x, adj1_blk, sparse=False, prompt_layers=prompt_layers)
                h2 = gcn(cat_x, adj2_blk, sparse=False, prompt_layers=prompt_layers)
                z1_g = _segment_means(h1, sizes)
                z2_g = _segment_means(h2, sizes)

                for local_i, sample_i in enumerate(chunk_indices):
                    z1_out[sample_i] = z1_g[local_i]
                    z2_out[sample_i] = z2_g[local_i]

    z1_out = F.normalize(egraphcl_proj(z1_out), p=2, dim=1)
    z2_out = F.normalize(egraphcl_proj(z2_out), p=2, dim=1)
    return z1_out, z2_out


def sample_pair_contrastive_loss(
    z1,
    z2,
    soft_pair_mask,
    temperature,
    negative_pair_mode='full',
    negative_topk=16,
    negative_random_ratio=0.25,
    min_negatives_per_row=1,
    return_stats=False,
):
    """Graph contrastive loss with optional negative-pair subsampling.

    negative_pair_mode:
      - full: use all valid negatives (original behavior)
      - topk: per anchor keep top-k hardest negatives by similarity
      - random: per anchor randomly keep a ratio of valid negatives
    """
    bsz = z1.size(0)
    if bsz <= 1:
        zero = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        stats = {'row_neg_min': 0, 'row_neg_mean': 0.0, 'row_neg_p10': 0.0}
        return (zero, stats) if return_stats else zero

    sim_logits = torch.matmul(z1, z2.t()) / float(temperature)
    sim = torch.exp(sim_logits)
    eye = torch.eye(bsz, device=z1.device, dtype=torch.bool)

    if soft_pair_mask is None:
        soft = torch.zeros((bsz, bsz), device=z1.device, dtype=torch.bool)
    else:
        soft = soft_pair_mask.to(z1.device)
        soft = soft > 0
    soft = soft & (~eye)

    neg_valid = (~soft) & (~eye)
    mode = str(negative_pair_mode).lower()

    if mode == 'topk':
        k = int(max(1, negative_topk))
        row_k = min(k, max(1, bsz - 1))
        masked_logits = sim_logits.masked_fill(~neg_valid, float('-inf'))
        top_idx = torch.topk(masked_logits, k=row_k, dim=1, largest=True).indices
        selected = torch.zeros_like(neg_valid)
        selected.scatter_(1, top_idx, True)
        neg_valid = selected & neg_valid

    elif mode == 'random':
        ratio = float(negative_random_ratio)
        ratio = max(0.0, min(1.0, ratio))
        row_k = int(round(max(1, bsz - 1) * ratio))
        row_k = max(1, min(row_k, max(1, bsz - 1)))
        rand_score = torch.rand_like(sim_logits)
        rand_score = rand_score.masked_fill(~neg_valid, float('inf'))
        keep_idx = torch.topk(rand_score, k=row_k, dim=1, largest=False).indices
        selected = torch.zeros_like(neg_valid)
        selected.scatter_(1, keep_idx, True)
        neg_valid = selected & neg_valid

    min_neg = max(1, int(min_negatives_per_row))
    row_neg_count = neg_valid.sum(dim=1)
    needs_refill = row_neg_count < min_neg
    if torch.any(needs_refill):
        refill_pool = (~eye) & (~soft) & (~neg_valid)
        refill_logits = sim_logits.masked_fill(~refill_pool, float('-inf'))
        refill_idx = torch.topk(refill_logits, k=min(min_neg, max(1, bsz - 1)), dim=1, largest=True).indices
        refill_mask = torch.zeros_like(neg_valid)
        refill_mask.scatter_(1, refill_idx, True)
        refill_mask = refill_mask & refill_pool
        row_selector = needs_refill.unsqueeze(1)
        neg_valid = torch.where(row_selector, neg_valid | refill_mask, neg_valid)

    row_neg_count = neg_valid.sum(dim=1)
    zero_neg_rows = row_neg_count <= 0
    if torch.any(zero_neg_rows):
        neg_valid = neg_valid.clone()
        neg_valid[zero_neg_rows] = (~eye)[zero_neg_rows]
        row_neg_count = neg_valid.sum(dim=1)

    pos_sim = sim[torch.arange(bsz, device=z1.device), torch.arange(bsz, device=z1.device)]
    neg_sum = (sim * neg_valid.float()).sum(dim=1)
    denom = pos_sim + neg_sum

    valid = denom > 0
    if torch.sum(valid).item() == 0:
        zero = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        stats = {
            'row_neg_min': int(row_neg_count.min().item()),
            'row_neg_mean': float(row_neg_count.float().mean().item()),
            'row_neg_p10': float(torch.quantile(row_neg_count.float(), 0.10).item()),
        }
        return (zero, stats) if return_stats else zero
    loss = -torch.log((pos_sim[valid] + 1e-12) / (denom[valid] + 1e-12)).mean()
    if not return_stats:
        return loss
    stats = {
        'row_neg_min': int(row_neg_count.min().item()),
        'row_neg_mean': float(row_neg_count.float().mean().item()),
        'row_neg_p10': float(torch.quantile(row_neg_count.float(), 0.10).item()),
    }
    return loss, stats


def build_sample_pair_soft_mask_from_embeddings(
    z,
    tau,
    strategy='fixed',
    quantile=0.9,
    min_tau=0.6,
    max_tau=0.98,
    return_stats=False,
):
    """Build soft mask from sample embeddings without cluster information.

    strategy:
      - fixed: use the provided tau directly
      - quantile: adapt tau from off-diagonal similarity quantile each batch
    """
    bsz = z.size(0)
    if bsz <= 1:
        if return_stats:
            return None, {
                'tau': float(tau),
                'mask_ratio': 0.0,
                'sim_mean': 0.0,
                'sim_p50': 0.0,
                'sim_p90': 0.0,
                'sim_p99': 0.0,
            }
        return None

    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.t())
    tau_val = float(tau)
    strategy = str(strategy).lower()

    if strategy == 'quantile':
        eye = torch.eye(bsz, device=z.device, dtype=torch.bool)
        off_diag = sim[~eye]
        if off_diag.numel() > 0:
            q = float(max(0.0, min(1.0, quantile)))
            tau_val = float(torch.quantile(off_diag.detach(), q).item())
            tau_val = max(float(min_tau), min(float(max_tau), tau_val))

    soft_mask = sim >= tau_val

    eye = torch.eye(bsz, device=z.device, dtype=torch.bool)
    soft_mask = soft_mask & (~eye)
    if not return_stats:
        return soft_mask.float()

    off_diag = sim[~eye]
    stats = {
        'tau': float(tau_val),
        'mask_ratio': float(soft_mask.float().mean().item()),
        'sim_mean': float(off_diag.mean().item()) if off_diag.numel() > 0 else 0.0,
        'sim_p50': float(torch.quantile(off_diag.detach(), 0.50).item()) if off_diag.numel() > 0 else 0.0,
        'sim_p90': float(torch.quantile(off_diag.detach(), 0.90).item()) if off_diag.numel() > 0 else 0.0,
        'sim_p99': float(torch.quantile(off_diag.detach(), 0.99).item()) if off_diag.numel() > 0 else 0.0,
    }
    return soft_mask.float(), stats


class EGraphCLCore:
    """Unified EGRAPHCL pipeline: encode -> mask -> pair-contrastive loss -> diagnostics."""

    def __init__(self):
        self.temperature = 0.2
        self.mask_tau = 0.8
        self.mask_strategy = 'fixed'
        self.mask_quantile = 0.9
        self.mask_min_tau = 0.6
        self.mask_max_tau = 0.98
        self.negative_pair_mode = 'full'
        self.negative_topk = 16
        self.negative_random_ratio = 0.25
        self.min_negatives_per_row = 1
        self.log_every = 20
        self.batched_chunk_graphs = 32
        self.prototype_assign_tau = 0.2
        self.prototype_consistency_loss_weight = 0.0
        self.prototype_balance_loss_weight = 0.0
        self.encoder_mode = 'batched'
        self.step = 0
        self.latest_stats = {}
        self.node_loss_weight = 0.0
        # 温度余弦退火（任务6）
        self.temperature_schedule = 'fixed'
        self.temperature_max = 0.5
        self.temperature_min = 0.05

    def configure(
        self,
        *,
        temperature,
        mask_tau,
        mask_strategy,
        mask_quantile,
        mask_min_tau,
        mask_max_tau,
        negative_pair_mode,
        negative_topk,
        negative_random_ratio,
        min_negatives_per_row,
        log_every,
        batched_chunk_graphs,
        prototype_assign_tau,
        prototype_consistency_loss_weight,
        prototype_balance_loss_weight,
        encoder_mode,
    ):
        self.temperature = float(temperature)
        self.mask_tau = float(mask_tau)
        self.mask_strategy = str(mask_strategy).lower()
        self.mask_quantile = float(mask_quantile)
        self.mask_min_tau = float(mask_min_tau)
        self.mask_max_tau = float(mask_max_tau)
        self.negative_pair_mode = str(negative_pair_mode).lower()
        self.negative_topk = int(negative_topk)
        self.negative_random_ratio = float(negative_random_ratio)
        self.min_negatives_per_row = int(min_negatives_per_row)
        self.log_every = max(1, int(log_every))
        self.batched_chunk_graphs = max(0, int(batched_chunk_graphs))
        self.prototype_assign_tau = max(1e-6, float(prototype_assign_tau))
        self.prototype_consistency_loss_weight = max(0.0, float(prototype_consistency_loss_weight))
        self.prototype_balance_loss_weight = max(0.0, float(prototype_balance_loss_weight))
        self.encoder_mode = str(encoder_mode).lower()

    def configure_temperature_schedule(self, schedule='fixed', temperature_max=0.5, temperature_min=0.05):
        """配置温度余弦退火调度（任务6）。"""
        self.temperature_schedule = str(schedule).lower()
        self.temperature_max = float(temperature_max)
        self.temperature_min = float(temperature_min)

    def get_temperature(self, global_step=0, total_steps=1):
        """计算当前温度（余弦退火或固定）。"""
        if self.temperature_schedule == 'cosine' and total_steps > 0:
            import math
            progress = float(global_step) / float(max(1, total_steps))
            progress = max(0.0, min(1.0, progress))
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.temperature_min + (self.temperature_max - self.temperature_min) * cosine_val
        return self.temperature
    def run(
        self,
        *,
        gcn,
        read,
        proj,
        proj_fea=None,
        proj_str=None,
        node_proj=None,
        node_loss_weight=None,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse,
        sample_dataset_ids,
        sample_prototype_ids,
        sample_prototype_weights,
        prompt_routing_mode,
        prompt_id_split_base,
        egraphcl_pair_soft_mask,
        combine_fn,
        global_step=0,
        total_steps=1,
    ):
        # 支持独立投影头（任务2），向后兼容单 proj 参数
        _proj_fea = proj_fea if proj_fea is not None else proj
        _proj_str = proj_str if proj_str is not None else proj
        # 温度余弦退火（任务6）
        current_temperature = self.get_temperature(global_step, total_steps)

        if self.encoder_mode == 'batched':
            z1_fea, z2_fea = compute_graph_pair_vectors_batched(
                gcn,
                _proj_fea,
                feature_prompt_layers,
                structure_prompt_layers,
                seq_list,
                adj_list,
                sparse=sparse,
                sample_dataset_ids=sample_dataset_ids,
                sample_prototype_ids=sample_prototype_ids,
                sample_prototype_weights=sample_prototype_weights,
                prompt_routing_mode=prompt_routing_mode,
                prompt_id_split_base=prompt_id_split_base,
                batched_chunk_graphs=self.batched_chunk_graphs,
                use_feature_prompt=True,
            )
            z1_str, z2_str = compute_graph_pair_vectors_batched(
                gcn,
                _proj_str,
                feature_prompt_layers,
                structure_prompt_layers,
                seq_list,
                adj_list,
                sparse=sparse,
                sample_dataset_ids=sample_dataset_ids,
                sample_prototype_ids=sample_prototype_ids,
                sample_prototype_weights=sample_prototype_weights,
                prompt_routing_mode=prompt_routing_mode,
                prompt_id_split_base=prompt_id_split_base,
                batched_chunk_graphs=self.batched_chunk_graphs,
                use_feature_prompt=False,
            )
        else:
            z1_fea, z2_fea = compute_graph_pair_vectors(
                gcn,
                read,
                _proj_fea,
                feature_prompt_layers,
                structure_prompt_layers,
                seq_list,
                adj_list,
                sparse=sparse,
                sample_dataset_ids=sample_dataset_ids,
                sample_prototype_ids=sample_prototype_ids,
                sample_prototype_weights=sample_prototype_weights,
                prompt_routing_mode=prompt_routing_mode,
                prompt_id_split_base=prompt_id_split_base,
                use_feature_prompt=True,
            )
            z1_str, z2_str = compute_graph_pair_vectors(
                gcn,
                read,
                _proj_str,
                feature_prompt_layers,
                structure_prompt_layers,
                seq_list,
                adj_list,
                sparse=sparse,
                sample_dataset_ids=sample_dataset_ids,
                sample_prototype_ids=sample_prototype_ids,
                sample_prototype_weights=sample_prototype_weights,
                prompt_routing_mode=prompt_routing_mode,
                prompt_id_split_base=prompt_id_split_base,
                use_feature_prompt=False,
            )

        if egraphcl_pair_soft_mask is None:
            z_mask = (z1_fea.detach() + z2_fea.detach()) / 2.0
            egraphcl_pair_soft_mask, mask_stats = build_sample_pair_soft_mask_from_embeddings(
                z_mask,
                self.mask_tau,
                strategy=self.mask_strategy,
                quantile=self.mask_quantile,
                min_tau=self.mask_min_tau,
                max_tau=self.mask_max_tau,
                return_stats=True,
            )
        else:
            mask_bool = egraphcl_pair_soft_mask > 0
            mask_stats = {
                'tau': float(self.mask_tau),
                'mask_ratio': float(mask_bool.float().mean().item()),
                'sim_mean': 0.0,
                'sim_p50': 0.0,
                'sim_p90': 0.0,
                'sim_p99': 0.0,
            }

        fea_loss, fea_stats = sample_pair_contrastive_loss(
            z1_fea,
            z2_fea,
            egraphcl_pair_soft_mask,
            current_temperature,
            negative_pair_mode=self.negative_pair_mode,
            negative_topk=self.negative_topk,
            negative_random_ratio=self.negative_random_ratio,
            min_negatives_per_row=self.min_negatives_per_row,
            return_stats=True,
        )
        str_loss, str_stats = sample_pair_contrastive_loss(
            z1_str,
            z2_str,
            egraphcl_pair_soft_mask,
            current_temperature,
            negative_pair_mode=self.negative_pair_mode,
            negative_topk=self.negative_topk,
            negative_random_ratio=self.negative_random_ratio,
            min_negatives_per_row=self.min_negatives_per_row,
            return_stats=True,
        )

        contrastive_loss = combine_fn(fea_loss, str_loss)
        total_loss = contrastive_loss

        # 节点级对比损失（任务4）
        node_loss_val = 0.0
        _node_loss_weight = node_loss_weight if node_loss_weight is not None else self.node_loss_weight
        if node_proj is not None and float(_node_loss_weight) > 0.0:
            # 从第一个样本提取节点 embedding（不经过 readout）
            try:
                seq0 = seq_list[0]
                adj0 = adj_list[0]
                h1_node = gcn(seq0[0], adj0[1], sparse, prompt_layers=None).squeeze(0)
                h2_node = gcn(seq0[0], adj0[2], sparse, prompt_layers=None).squeeze(0)
                n_loss = node_contrastive_loss(h1_node, h2_node, self.temperature, node_proj)
                total_loss = total_loss + float(_node_loss_weight) * n_loss
                node_loss_val = float(n_loss.detach().item())
            except Exception:
                pass

        proto_reg = torch.tensor(0.0, device=z1_fea.device, dtype=z1_fea.dtype)
        proto_stats = {
            'proto_consistency_loss': 0.0,
            'proto_balance_loss': 0.0,
            'proto_entropy': 0.0,
        }
        if str(prompt_routing_mode).lower() == 'prototype' and sample_prototype_weights is not None:
            q_prior = _build_proto_weight_matrix(
                sample_prototype_weights,
                prompt_bank_size=len(feature_prompt_layers),
                device=z1_fea.device,
            )
            proto_reg, proto_stats = _prototype_regularization(
                z1_fea,
                z2_fea,
                q_prior,
                tau=self.prototype_assign_tau,
                w_consistency=self.prototype_consistency_loss_weight,
                w_balance=self.prototype_balance_loss_weight,
            )
            total_loss = total_loss + proto_reg

        self.step += 1
        self.latest_stats = {
            'mask_ratio': float(mask_stats['mask_ratio']),
            'tau': float(mask_stats['tau']),
            'sim_mean': float(mask_stats['sim_mean']),
            'sim_p90': float(mask_stats['sim_p90']),
            'sim_p99': float(mask_stats['sim_p99']),
            'fea_loss': float(fea_loss.detach().item()),
            'str_loss': float(str_loss.detach().item()),
            'contrastive_loss': float(contrastive_loss.detach().item()),
            'total_loss': float(total_loss.detach().item()),
            'fea_row_neg_min': int(fea_stats['row_neg_min']),
            'fea_row_neg_mean': float(fea_stats['row_neg_mean']),
            'str_row_neg_min': int(str_stats['row_neg_min']),
            'str_row_neg_mean': float(str_stats['row_neg_mean']),
            'proto_consistency_loss': float(proto_stats['proto_consistency_loss']),
            'proto_balance_loss': float(proto_stats['proto_balance_loss']),
            'proto_entropy': float(proto_stats['proto_entropy']),
            'proto_regularization': float(proto_reg.detach().item()),
            'node_loss': node_loss_val,
            'current_temperature': float(current_temperature),
        }

        if (self.step % self.log_every) == 0:
            print(
                '[egraphcl] mask/loss stats '
                f"tau={self.latest_stats['tau']:.4f} "
                f"mask_ratio={self.latest_stats['mask_ratio']:.4f} "
                f"sim_p90={self.latest_stats['sim_p90']:.4f} "
                f"neg_min(fea/str)={self.latest_stats['fea_row_neg_min']}/"
                f"{self.latest_stats['str_row_neg_min']}"
            )

        return total_loss, self.latest_stats
