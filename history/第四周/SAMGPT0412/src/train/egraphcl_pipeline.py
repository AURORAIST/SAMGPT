import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
    use_feature_prompt=True,
):
    """Build sample-level graph embeddings for two augmented graph views."""
    if sample_dataset_ids is not None and torch.is_tensor(sample_dataset_ids):
        sample_dataset_ids = sample_dataset_ids.detach().cpu().tolist()

    z1_list = []
    z2_list = []
    for sample_i, (seq, adj) in enumerate(zip(seq_list, adj_list)):
        dataset_id = sample_i if sample_dataset_ids is None else int(sample_dataset_ids[sample_i])
        if dataset_id < 0 or dataset_id >= len(feature_prompt_layers):
            raise IndexError(
                f"dataset_id={dataset_id} out of range for prompt layers (size={len(feature_prompt_layers)})"
            )

        fea_pretext = feature_prompt_layers[dataset_id]
        str_layers = structure_prompt_layers[dataset_id]

        if use_feature_prompt:
            seq_in = fea_pretext(seq[0])
            prompt_layers = None
        else:
            seq_in = seq[0]
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


def sample_pair_contrastive_loss(
    z1,
    z2,
    soft_pair_mask,
    temperature,
    negative_pair_mode='full',
    negative_topk=16,
    negative_random_ratio=0.25,
):
    """Graph contrastive loss with optional negative-pair subsampling.

    negative_pair_mode:
      - full: use all valid negatives (original behavior)
      - topk: per anchor keep top-k hardest negatives by similarity
      - random: per anchor randomly keep a ratio of valid negatives
    """
    bsz = z1.size(0)
    if bsz <= 1:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

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
        selected = torch.zeros_like(neg_valid)
        for i in range(bsz):
            valid_idx = torch.nonzero(neg_valid[i], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue
            row_k = min(k, int(valid_idx.numel()))
            row_logits = sim_logits[i, valid_idx]
            top_idx = torch.topk(row_logits, k=row_k, largest=True).indices
            keep_idx = valid_idx[top_idx]
            selected[i, keep_idx] = True
        neg_valid = selected

    elif mode == 'random':
        ratio = float(negative_random_ratio)
        ratio = max(0.0, min(1.0, ratio))
        selected = torch.zeros_like(neg_valid)
        for i in range(bsz):
            valid_idx = torch.nonzero(neg_valid[i], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue
            row_k = max(1, int(round(valid_idx.numel() * ratio)))
            row_k = min(row_k, int(valid_idx.numel()))
            rand_perm = torch.randperm(valid_idx.numel(), device=z1.device)[:row_k]
            keep_idx = valid_idx[rand_perm]
            selected[i, keep_idx] = True
        neg_valid = selected

    row_neg_count = neg_valid.sum(dim=1)
    zero_neg_rows = row_neg_count <= 0
    if torch.any(zero_neg_rows):
        neg_valid = neg_valid.clone()
        neg_valid[zero_neg_rows] = (~eye)[zero_neg_rows]

    pos_sim = sim[torch.arange(bsz, device=z1.device), torch.arange(bsz, device=z1.device)]
    neg_sum = (sim * neg_valid.float()).sum(dim=1)
    denom = pos_sim + neg_sum

    valid = denom > 0
    if torch.sum(valid).item() == 0:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    loss = -torch.log((pos_sim[valid] + 1e-12) / (denom[valid] + 1e-12)).mean()
    return loss


def build_sample_pair_soft_mask_from_embeddings(z, tau):
    """Build soft mask from sample embeddings without cluster information."""
    bsz = z.size(0)
    if bsz <= 1:
        return None

    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.t())
    soft_mask = sim >= float(tau)

    eye = torch.eye(bsz, device=z.device, dtype=torch.bool)
    soft_mask = soft_mask & (~eye)
    return soft_mask.float()
