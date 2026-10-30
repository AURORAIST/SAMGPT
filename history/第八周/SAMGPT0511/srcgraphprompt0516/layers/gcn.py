import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        # print("act",type(self.act))
        # print("fc",self.fc.weight)
        # print("fc",self.fc.weight.shape)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Expected input shapes:
    #  - seq: (B, N, F) or (N, F) where B is batch size (current implementation supports B==1)
    #  - adj: for sparse=True -> torch.sparse_coo_tensor shape [N, N]
    #         for sparse=False -> dense tensor shape [1, N, N] or [N, N]
    def forward(self, input, sparse=True):
        # print("input",input)
        seq = input[0]#.to(input.device)
        adj = input[1]#.to(input.device)

        # Track if input was 2D (no batch dimension)
        was_2d = seq.dim() == 2
        
        # Add batch dimension if missing
        if was_2d:
            seq = seq.unsqueeze(0)  # Convert [N, F] to [1, N, F]

        # Validate shapes
        if not torch.is_tensor(seq):
            raise TypeError(f"GCN.forward expected seq to be a torch.Tensor, got {type(seq)}")
        if seq.dim() != 3:
            raise ValueError(f"GCN.forward expects seq with shape [B, N, F], got {tuple(seq.shape)}")
        if seq.shape[0] != 1:
            raise ValueError(f"GCN.forward currently supports batch size 1 only (got B={seq.shape[0]})")
        
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        seq_fts = self.fc(seq)  # [1, N, F']
        
        # Squeeze batch dimension for sparse operations
        seq_fts_2d = seq_fts.squeeze(0)  # [N, F']
        
        if sparse:
            out = torch.spmm(adj, seq_fts_2d)  # [N, N] x [N, F'] -> [N, F']
        else:
            # print("adj",adj.shape)
            # print("seqft",seq_fts.shape)
            out = torch.mm(adj.squeeze(dim=0), seq_fts_2d)
        
        if self.bias is not None:
            out += self.bias

        # print("out",out)
        # print("act",self.act)

        return self.act(out)
        # return out.type(torch.float)
