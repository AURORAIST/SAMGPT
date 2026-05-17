import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(
        self,
        gcn,
        seq1,
        seq2,
        adj,
        sparse,
        msk,
        samp_bias1,
        samp_bias2,
        return_reliability=False,
        reliability_mode='embedding',
    ):
        h_1 = gcn(seq1, adj, sparse)

        h_3 = h_1 * self.prompt

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)

        if return_reliability:
            return ret, self._embedding_reliability_weights(h_3, h_4)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    @staticmethod
    def _to_2d_embedding(emb):
        if emb.dim() == 3 and emb.shape[0] == 1:
            return emb.squeeze(0)
        return emb

    def _embedding_reliability_weights(self, h_pos, h_neg):
        with torch.no_grad():
            h_pos = self._to_2d_embedding(h_pos).detach()
            h_neg = self._to_2d_embedding(h_neg).detach()
            pos_weight = torch.ones(h_pos.shape[0], device=h_pos.device)
            if h_pos.shape == h_neg.shape:
                dot = (h_pos * h_neg).sum(dim=1)
                pos_norm = (h_pos * h_pos).sum(dim=1).clamp_min(1e-12).rsqrt()
                neg_norm = (h_neg * h_neg).sum(dim=1).clamp_min(1e-12).rsqrt()
                neg_cos = dot * pos_norm * neg_norm
                neg_weight = torch.sigmoid(1.0 - neg_cos).clamp_(0.05, 1.0)
            else:
                neg_weight = torch.ones(h_pos.shape[0], device=h_pos.device)
            return torch.cat([pos_weight, neg_weight], dim=0).unsqueeze(0)

    # # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = self.gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
    #
