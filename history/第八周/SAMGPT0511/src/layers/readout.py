import torch
import torch.nn as nn


# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    """平均读出（readout）层。

    默认保持原来的全图平均读出；当传入邻接矩阵 `adj` 时，改为对每个节点的
    2-hop 邻域做局部 readout，返回每个节点自己的上下文向量。

    Shapes:
        seq: [B, N, F]
        msk: [B, N] 或 None
        adj: [N, N] 或 sparse [N, N] 或 None
        return: adj is None 时 [B, F]；否则 [B, N, F]
    """

    def __init__(self):
        super(AvgReadout, self).__init__()
        self._debug_printed = False
        self._two_hop_adj_cache = {}

    def forward(self, seq, msk=None, adj=None):
        """计算 readout 向量。

        Args:
            seq: 节点表示序列，形状 [batch, nodes, features]
            msk: 节点掩码，形状 [batch, nodes]；为 None 表示不使用掩码
            adj: 邻接矩阵；为 None 时使用原始全图平均

        Returns:
            全图或每节点 2-hop 子图表示
        """
        if adj is not None:
            return self._two_hop_readout(seq, msk, adj)

        if msk is None:
            # 沿节点维（dim=1）做均值
            return torch.mean(seq, 1)
        else:
            # 扩展 mask 以便与 seq 在特征维上广播相乘
            # msk: [B,N] -> [B,N,1]
            msk = torch.unsqueeze(msk, -1)
            # masked average: sum(seq*msk)/sum(msk)
            # 其中 sum(msk) 是每个样本被选中节点的数量（广播到特征维）
            return torch.sum(seq * msk, 1) / torch.sum(msk).clamp_min(1.0)

    def _two_hop_readout(self, seq, msk, adj):
        """用 A^2 对节点表示做 2-hop 局部平均。

        训练代码里传入的邻接通常已经包含 self-loop，因此 A^2 覆盖中心节点、
        1-hop 和 2-hop；如果没有 self-loop，则严格是 2-hop 路径聚合。
        """
        squeeze_batch = False
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
            squeeze_batch = True

        if msk is not None:
            if msk.dim() == 1:
                msk = msk.unsqueeze(0)
            seq = seq * msk.unsqueeze(-1)

        two_hop_adj = self._get_two_hop_adj(adj, seq)
        if two_hop_adj.is_sparse:
            out = self._readout_sparse(seq, msk, two_hop_adj)
        else:
            out = self._readout_dense(seq, msk, two_hop_adj)

        if squeeze_batch:
            out = out.squeeze(0)
        self._log_two_hop_readout(seq, msk, adj, two_hop_adj, out)
        return out

    def _log_two_hop_readout(self, seq, msk, adj, two_hop_adj, out):
        if self._debug_printed:
            return

        self._debug_printed = True
        nnz = adj._nnz() if adj.is_sparse else int(torch.count_nonzero(adj).item())
        two_hop_nnz = two_hop_adj._nnz() if two_hop_adj.is_sparse else int(torch.count_nonzero(two_hop_adj).item())
        print(
            "[AvgReadout] using 2-hop subgraph readout | "
            f"seq={tuple(seq.shape)} | "
            f"adj={tuple(adj.shape)} | "
            f"sparse={adj.is_sparse} | "
            f"nnz={nnz} | "
            f"two_hop_nnz={two_hop_nnz} | "
            f"mask={None if msk is None else tuple(msk.shape)} | "
            f"out={tuple(out.shape)}"
        )

    def _get_two_hop_adj(self, adj, seq):
        cache_key = self._adj_cache_key(adj)
        cached = self._two_hop_adj_cache.get(cache_key)
        if cached is not None and cached.device == seq.device and cached.dtype == seq.dtype:
            return cached

        if adj.is_sparse:
            two_hop_adj = self._build_two_hop_sparse_adj(adj, seq)
        else:
            two_hop_adj = self._build_two_hop_dense_adj(adj, seq)

        self._two_hop_adj_cache[cache_key] = two_hop_adj
        return two_hop_adj

    def _adj_cache_key(self, adj):
        if adj.is_sparse:
            adj = adj.coalesce()
            return (
                'sparse',
                adj.device.type,
                adj.device.index,
                tuple(adj.shape),
                adj.indices().data_ptr(),
                adj.values().data_ptr(),
            )

        dense_adj = adj.squeeze(0) if adj.dim() == 3 else adj
        return (
            'dense',
            dense_adj.device.type,
            dense_adj.device.index,
            tuple(dense_adj.shape),
            dense_adj.data_ptr(),
        )

    def _build_two_hop_sparse_adj(self, adj, seq):
        adj = adj.coalesce().to(seq.device)
        values = torch.ones_like(adj.values(), dtype=seq.dtype, device=seq.device)
        one_hop_adj = torch.sparse_coo_tensor(adj.indices(), values, adj.shape, device=seq.device).coalesce()
        two_hop_adj = torch.sparse.mm(one_hop_adj, one_hop_adj).coalesce()
        values = torch.ones_like(two_hop_adj.values(), dtype=seq.dtype, device=seq.device)
        return torch.sparse_coo_tensor(two_hop_adj.indices(), values, two_hop_adj.shape, device=seq.device).coalesce()

    def _build_two_hop_dense_adj(self, adj, seq):
        adj = adj.to(device=seq.device)
        if adj.dim() == 3:
            adj = adj.squeeze(0)
        one_hop_adj = (adj != 0).to(dtype=seq.dtype)
        return (torch.matmul(one_hop_adj, one_hop_adj) > 0).to(dtype=seq.dtype)

    def _readout_sparse(self, seq, msk, two_hop_adj):
        outs = []
        for batch_idx in range(seq.shape[0]):
            two_hop = torch.sparse.mm(two_hop_adj, seq[batch_idx])

            if msk is None:
                denom_src = torch.ones(seq.shape[1], 1, dtype=seq.dtype, device=seq.device)
            else:
                denom_src = msk[batch_idx].to(dtype=seq.dtype, device=seq.device).unsqueeze(-1)

            denom = torch.sparse.mm(two_hop_adj, denom_src).clamp_min(1e-12)
            outs.append(two_hop / denom)

        return torch.stack(outs, dim=0)

    def _readout_dense(self, seq, msk, two_hop_adj):
        two_hop = torch.matmul(two_hop_adj, seq)

        if msk is None:
            denom_src = torch.ones(seq.shape[0], seq.shape[1], 1, dtype=seq.dtype, device=seq.device)
        else:
            denom_src = msk.to(dtype=seq.dtype, device=seq.device).unsqueeze(-1)

        denom = torch.matmul(two_hop_adj, denom_src).clamp_min(1e-12)
        return two_hop / denom
