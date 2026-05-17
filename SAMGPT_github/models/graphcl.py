import torch
import torch.nn as nn
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

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type, prompt_layers = None):

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
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    # Detach the return variables
    # def embed(self, seq, adj, sparse, msk):
    #     h_1 = gcn(seq, adj, sparse)
    #     c = self.read(h_1, msk)
    #
    #     return h_1.detach(), c.detach()
