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

        self.reset_parameters()

    def encode(self, gcn, seq1, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, aug_type, prompt_layers=None):
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

        return h_0, h_1, h_3, c_1, c_3

    def nodecl_logits(self, encoded, batch_size, temperature=0.2, pos_idx=None):
        _, h_1, h_3, _, _ = encoded
        z1 = F.normalize(h_1.squeeze(0), dim=-1)
        z2 = F.normalize(h_3.squeeze(0), dim=-1)

        node_num = z1.shape[0]
        if batch_size is None or batch_size <= 0:
            batch_size = node_num
        batch_size = min(batch_size, node_num)

        if pos_idx is None:
            pos_idx = torch.randperm(node_num, device=z1.device)[:batch_size]

        logits12 = torch.matmul(z1[pos_idx], z2.t()) / temperature
        logits21 = torch.matmul(z2[pos_idx], z1.t()) / temperature
        return logits12, logits21, pos_idx

    def cross_domain_logits(self, encoded, neg_pool, batch_size, pos_idx=None, neg_idx=None):
        h_0, _, _, c_1, c_3 = encoded
        h_pos = h_0.squeeze(0)
        if neg_pool.dim() == 3:
            neg_pool = neg_pool.squeeze(0)

        node_num = h_pos.shape[0]
        neg_num = neg_pool.shape[0]
        if batch_size is None or batch_size <= 0:
            batch_size = node_num
        batch_size = min(batch_size, node_num)

        if pos_idx is None:
            pos_idx = torch.randperm(node_num, device=h_pos.device)[:batch_size]
        if neg_idx is None:
            neg_idx = torch.randint(0, neg_num, (batch_size,), device=h_pos.device)
        h_pos = h_pos[pos_idx].unsqueeze(0)
        h_neg = neg_pool[neg_idx].unsqueeze(0)

        ret1 = self._disc_logits(c_1, h_pos, h_neg)
        ret2 = self._disc_logits(c_3, h_pos, h_neg)
        return ret1 + ret2

    def _disc_logits(self, c_vec, h_pos, h_neg):
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

        W = self.disc.f_k.weight.squeeze(0)
        bias = self.disc.f_k.bias
        if bias is None:
            bias = 0.0
        else:
            bias = bias.squeeze(0)

        sc_pos = (torch.matmul(h_pos, W) * c_expand).sum(dim=-1) + bias
        sc_neg = (torch.matmul(h_neg, W) * c_expand).sum(dim=-1) + bias
        return torch.cat((sc_pos, sc_neg), dim=1)

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type, prompt_layers = None, neg_pool=None, batch_size=None):
        encoded = self.encode(gcn, seq1, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, aug_type, prompt_layers)
        if neg_pool is not None:
            return self.cross_domain_logits(encoded, neg_pool, batch_size)

        h_2 = gcn(seq2, adj, sparse, prompt_layers)
        ret1 = self._disc_logits(encoded[3], encoded[0], h_2)
        ret2 = self._disc_logits(encoded[4], encoded[0], h_2)

        return ret1 + ret2

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
