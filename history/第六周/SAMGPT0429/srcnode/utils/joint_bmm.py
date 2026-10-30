import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import logging


logger = logging.getLogger(__name__)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_fn_posterior_from_pairs(sim, tuples_idx, adj=None, seed=0, n_components=2):
    """
    使用 2D GaussianMixture 在每个 anchor-候选对上拟合 (s,d) 的混合分布并返回 p(FN|s,d)。

    Args:
        sim: Tensor[N, K] 的余弦相似度（包含 pos 在第0列）
        tuples_idx: LongTensor[N, K] 的索引对应候选节点
        adj: 可选的邻接矩阵（torch sparse 或 dense），若提供则计算 Jaccard 作为结构相似度 d

    Returns:
        p_fn: numpy array shape (N, K) 每个对属于 FN 组件的后验概率
    """
    device = sim.device if isinstance(sim, torch.Tensor) else None
    N, K = sim.shape

    logger.info(
        "[joint_bmm] start posterior fit: sim_shape=%s, tuples_shape=%s, adj=%s",
        tuple(sim.shape),
        tuple(tuples_idx.shape),
        "yes" if adj is not None else "no",
    )

    # 计算结构相似度 d
    if adj is None:
        d = np.zeros((N, K), dtype=np.float32)
    else:
        if isinstance(adj, torch.Tensor) and adj.dim() > 2:
            adj = adj[0]
        # 将邻接转换为 dense bool（在大多数预训练图上可行）
        if isinstance(adj, torch.Tensor) and adj.is_sparse:
            A = adj.to_dense().bool()
        elif isinstance(adj, torch.Tensor):
            A = adj.bool()
        else:
            A = torch.tensor(adj).bool()

        A = A.cpu()
        deg = A.sum(dim=1).float()

        d = np.zeros((N, K), dtype=np.float32)
        tuples_cpu = tuples_idx.detach().cpu() if isinstance(tuples_idx, torch.Tensor) else torch.tensor(tuples_idx)

        for i in range(N):
            ai = A[i]  # bool row
            cand = tuples_cpu[i]  # K
            Acand = A[cand]
            inter = (Acand & ai).sum(dim=1).float()
            union = (ai.sum() + Acand.sum(dim=1).float() - inter).float()
            # jaccard
            j = torch.zeros_like(inter)
            mask = union > 0
            j[mask] = inter[mask] / union[mask]
            d[i, :] = j.numpy()

    s = _to_numpy(sim)

    X = np.stack([s.reshape(-1), d.reshape(-1)], axis=1)

    # Fit GMM
    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
        gmm.fit(X)
        resp = gmm.predict_proba(X)  # (N*K, n_components)
        logger.info(
            "[joint_bmm] gmm fitted successfully: n_samples=%d, n_components=%d",
            X.shape[0],
            n_components,
        )
    except Exception:
        # If GMM fails (degenerate), fall back to simple thresholding on s
        s_flat = X[:, 0]
        median = np.median(s_flat)
        resp = np.vstack([1 - (s_flat > median).astype(float), (s_flat > median).astype(float)]).T
        logger.warning("[joint_bmm] gmm fit failed, fallback to similarity thresholding")

    # Determine which component is FN: choose component with larger mean similarity
    comp_means = []
    for c in range(resp.shape[1]):
        comp_means.append(X[:, 0].dot(resp[:, c]) / (resp[:, c].sum() + 1e-12))
    idx_fn = int(np.argmax(comp_means))

    p_fn = resp[:, idx_fn].reshape(N, K).astype(np.float32)
    logger.info(
        "[joint_bmm] posterior summary: fn_component=%d, p_fn[min=%.4f, mean=%.4f, max=%.4f]",
        idx_fn,
        float(p_fn.min()),
        float(p_fn.mean()),
        float(p_fn.max()),
    )
    return p_fn
