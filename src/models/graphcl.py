import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, AvgReadout, Discriminator, Discriminator2
import pdb


class GraphCL(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCL, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1,n_h), requires_grad=True)
        self._descriptor_cache = {}

        self.reset_parameters()

    def forward(
        self,
        gcn,
        seq1,
        seq2,
        seq3,
        seq4,
        adj,
        aug_adj1,
        aug_adj2,
        sparse,
        msk,
        samp_bias1,
        samp_bias2,
        aug_type,
        prompt_layers=None,
        return_reliability=False,
        reliability_mode='embedding',
    ):

        #print('seq1', seq1.shape)
        #print('adj', adj.shape)
        h_0 = gcn(seq1, adj, sparse)

        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse, prompt_layers)
            h_3 = gcn(seq1, aug_adj2, sparse, prompt_layers)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse, prompt_layers)
            h_3 = gcn(seq4, adj, sparse, prompt_layers)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse, prompt_layers)
            h_3 = gcn(seq4, aug_adj2, sparse, prompt_layers)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse, prompt_layers)

        # ---- Robust discriminator score computation ----
        # Some CUDA builds hit CUBLAS_STATUS_INVALID_VALUE inside nn.Bilinear even when shapes are valid.
        # Compute bilinear score explicitly: score = x^T W y + b
        def _disc_logits(c_vec, h_pos, h_neg, disc_module: Discriminator):
            # c_vec: [B, F]; h_pos/h_neg: [B, N, F]
            if h_pos.dim() == 2:
                h_pos = h_pos.unsqueeze(0)
            if h_neg.dim() == 2:
                h_neg = h_neg.unsqueeze(0)
            if c_vec.dim() == 1:
                c_vec = c_vec.unsqueeze(0)

            bsz, n, f = h_pos.shape
            c_expand = c_vec.unsqueeze(1).expand(bsz, n, f).contiguous()
            h_pos = h_pos.contiguous()
            h_neg = h_neg.contiguous()

            # W: [out=1, in1=f, in2=f]
            W = disc_module.f_k.weight.squeeze(0)  # [f, f]
            bias = disc_module.f_k.bias
            if bias is None:
                bias = 0.0
            else:
                bias = bias.squeeze(0)

            # sc = sum_{i,j} h[i]*W[i,j]*c[j]
            # Implement as (h @ W) * c then sum over feature dim.
            sc_pos = (torch.matmul(h_pos, W) * c_expand).sum(dim=-1) + bias
            sc_neg = (torch.matmul(h_neg, W) * c_expand).sum(dim=-1) + bias

            if samp_bias1 is not None:
                sc_pos = sc_pos + samp_bias1
            if samp_bias2 is not None:
                sc_neg = sc_neg + samp_bias2

            return torch.cat((sc_pos, sc_neg), dim=1)

        ret1 = _disc_logits(c_1, h_0, h_2, self.disc)
        ret2 = _disc_logits(c_3, h_0, h_2, self.disc)
        # ---- end ----

        ret = ret1 + ret2
        if return_reliability:
            weights = self._reliability_weights(
                seq1,
                seq2,
                seq3,
                seq4,
                adj,
                aug_adj1,
                aug_adj2,
                h_0,
                h_1,
                h_3,
                h_2,
                aug_type,
                reliability_mode,
            )
            return ret, weights
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    @staticmethod
    def _to_2d_seq(seq):
        if seq.dim() == 3 and seq.shape[0] == 1:
            return seq.squeeze(0)
        return seq

    @staticmethod
    def _standardize(x):
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True, unbiased=False) + 1e-6)

    def _adj_edges(self, adj):
        if adj.is_sparse:
            adj = adj.coalesce()
            row, col = adj.indices()
            num_nodes = adj.shape[0]
        else:
            dense_adj = adj.squeeze(0) if adj.dim() == 3 else adj
            row, col = torch.nonzero(dense_adj, as_tuple=True)
            num_nodes = dense_adj.shape[0]
        keep = row != col
        return row[keep].detach().cpu(), col[keep].detach().cpu(), num_nodes

    def _structure_descriptor(self, adj, device):
        row, col, num_nodes = self._adj_edges(adj)
        if row.numel() == 0:
            edge_signature = (0, 0, 0)
        else:
            edge_hash = torch.remainder((row.long() * 1000003 + col.long()).sum(), 2147483647).item()
            edge_signature = (int(row.sum().item()), int(col.sum().item()), int(edge_hash))
        cache_key = (str(device), num_nodes, row.numel(), edge_signature)
        if cache_key in self._descriptor_cache:
            return self._descriptor_cache[cache_key]

        neighbors = [set() for _ in range(num_nodes)]
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].add(c)

        degree_vals = []
        clustering_vals = []
        ego_density_vals = []
        for i, neigh in enumerate(neighbors):
            deg = len(neigh)
            degree_vals.append(deg)

            if deg < 2:
                clustering_vals.append(0.0)
                ego_density_vals.append(0.0 if deg == 0 else 1.0)
                continue

            inner_twice = 0
            for u in neigh:
                inner_twice += len(neighbors[u].intersection(neigh))
            inner_edges = inner_twice / 2.0
            clustering_vals.append(inner_twice / (deg * (deg - 1.0)))

            ego_nodes = deg + 1
            ego_edges = deg + inner_edges
            ego_density_vals.append((2.0 * ego_edges) / (ego_nodes * (ego_nodes - 1.0)))

        degree = torch.tensor(degree_vals, dtype=torch.float32, device=device).unsqueeze(1)
        clustering = torch.tensor(clustering_vals, dtype=torch.float32, device=device).unsqueeze(1)
        ego_density = torch.tensor(ego_density_vals, dtype=torch.float32, device=device).unsqueeze(1)
        pe = self._laplacian_pe(row, col, num_nodes, device)

        struct = torch.cat([torch.log1p(degree), clustering, ego_density, pe], dim=1)
        struct = self._standardize(struct)
        self._descriptor_cache[cache_key] = struct
        return struct

    def _laplacian_pe(self, row, col, num_nodes, device):
        # Full eigendecomposition is only reasonable for small graphs. Large graphs keep a zero PE
        # so descriptor mode remains usable on Pubmed/Amazon-style datasets.
        if num_nodes > 2000:
            return torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
        try:
            dense_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
            dense_adj[row, col] = 1.0
            dense_adj = torch.maximum(dense_adj, dense_adj.t())
            deg = dense_adj.sum(dim=1)
            inv_sqrt = deg.clamp_min(1e-12).rsqrt()
            lap = torch.eye(num_nodes) - inv_sqrt[:, None] * dense_adj * inv_sqrt[None, :]
            _, eigvecs = torch.linalg.eigh(lap)
            pe = eigvecs[:, 1:2] if eigvecs.shape[1] > 1 else eigvecs[:, :1]
            return pe.to(device)
        except Exception:
            return torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)

    def _context_descriptor(self, seq, adj):
        row, col, num_nodes = self._adj_edges(adj)
        row = torch.cat([row, torch.arange(num_nodes)])
        col = torch.cat([col, torch.arange(num_nodes)])
        indices = torch.stack([row, col], dim=0).to(seq.device)
        values = torch.ones(indices.shape[1], dtype=seq.dtype, device=seq.device)
        closed_adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=seq.device).coalesce()
        denom = torch.sparse.sum(closed_adj, dim=1).to_dense().unsqueeze(1).clamp_min(1.0)
        return torch.sparse.mm(closed_adj, seq) / denom

    def _node_descriptor(self, seq, adj):
        """Build psi_i = [struct || context] used by reliability weighting."""
        seq = self._to_2d_seq(seq).float()
        struct = self._structure_descriptor(adj, seq.device)
        context = self._context_descriptor(seq, adj)
        context = F.normalize(context, p=2, dim=1, eps=1e-6)
        return torch.cat([struct, context], dim=1)

    def _safe_pair_cos(self, left, right):
        if left.shape != right.shape:
            return None
        return F.cosine_similarity(left, right, dim=1, eps=1e-6)

    @staticmethod
    def _to_2d_embedding(emb):
        if emb.dim() == 3 and emb.shape[0] == 1:
            return emb.squeeze(0)
        return emb

    def _embedding_reliability_weights(self, h_0, h_1, h_3, h_2):
        with torch.no_grad():
            h0 = self._to_2d_embedding(h_0).detach()
            h1 = self._to_2d_embedding(h_1).detach()
            h3 = self._to_2d_embedding(h_3).detach()
            h2 = self._to_2d_embedding(h_2).detach()

            def _fast_row_cos(left, right):
                if left.shape != right.shape:
                    return None
                dot = (left * right).sum(dim=1)
                left_norm = (left * left).sum(dim=1).clamp_min(1e-12).rsqrt()
                right_norm = (right * right).sum(dim=1).clamp_min(1e-12).rsqrt()
                return dot * left_norm * right_norm

            pos_cos = _fast_row_cos(h1, h3)
            if pos_cos is None:
                pos_weight = torch.ones(h0.shape[0], device=h0.device)
            else:
                pos_weight = ((pos_cos + 1.0) * 0.5).clamp_(0.05, 1.0)

            neg_cos = _fast_row_cos(h0, h2)
            if neg_cos is None:
                neg_weight = torch.ones(h0.shape[0], device=h0.device)
            else:
                neg_weight = torch.sigmoid(1.0 - neg_cos).clamp_(0.05, 1.0)

            return torch.cat([pos_weight, neg_weight], dim=0).unsqueeze(0)

    def _reliability_weights(
        self,
        seq1,
        seq2,
        seq3,
        seq4,
        adj,
        aug_adj1,
        aug_adj2,
        h_0,
        h_1,
        h_3,
        h_2,
        aug_type,
        reliability_mode='embedding',
    ):
        """Return BCE weights [B, 2N] from PDF positive/negative reliability."""
        if reliability_mode == 'embedding':
            return self._embedding_reliability_weights(h_0, h_1, h_3, h_2)

        with torch.no_grad():
            base_desc = self._node_descriptor(seq1, adj)
            neg_desc = self._node_descriptor(seq2, adj)

            if aug_type == 'edge':
                view1_desc = self._node_descriptor(seq1, aug_adj1)
                view2_desc = self._node_descriptor(seq1, aug_adj2)
            elif aug_type == 'mask':
                view1_desc = self._node_descriptor(seq3, adj)
                view2_desc = self._node_descriptor(seq4, adj)
            else:
                view1_desc = self._node_descriptor(seq3, aug_adj1)
                view2_desc = self._node_descriptor(seq4, aug_adj2)

            pos_cos = self._safe_pair_cos(view1_desc, view2_desc)
            if pos_cos is None:
                pos_weight = torch.ones(base_desc.shape[0], device=base_desc.device)
            else:
                pos_weight = ((pos_cos + 1.0) * 0.5).clamp(0.05, 1.0)

            desc_cos = self._safe_pair_cos(base_desc, neg_desc)
            if desc_cos is None:
                neg_weight = torch.ones(base_desc.shape[0], device=base_desc.device)
            else:
                h0 = h_0.squeeze(0) if h_0.dim() == 3 and h_0.shape[0] == 1 else h_0
                h2 = h_2.squeeze(0) if h_2.dim() == 3 and h_2.shape[0] == 1 else h_2
                z_cos = self._safe_pair_cos(h0, h2)
                if z_cos is None:
                    z_cos = torch.zeros_like(desc_cos)
                neg_weight = torch.sigmoid(1.0 - z_cos - desc_cos).clamp(0.05, 1.0)

            weights = torch.cat([pos_weight, neg_weight], dim=0).unsqueeze(0)
            return weights

    # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
