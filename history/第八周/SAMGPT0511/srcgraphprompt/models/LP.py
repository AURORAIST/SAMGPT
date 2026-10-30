import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl
# import dgl.function as fn
import copy
from functools import partial
# from dgl.nn.pytorch.conv import RelGraphConv
# from basemodel import GraphAdjModel
import numpy as np
import tqdm

class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.n_in = n_in
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        # 将 prompt 定义为可训练参数，且用于缩放输出
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()
    def forward(self,gcn,seq,adj,sparse,prompt_layers=None):
        ret = gcn(seq,adj,sparse,True,prompt_layers)
        # ret = h_1
        # ret = h_1 * self.prompt
        # print("ret1",ret)
        # 使定义的 prompt 实际影响输出（按特征维缩放）
        ret = self.sigm(ret.squeeze(dim=0))
        try:
            ret = ret * self.prompt
        except Exception:
            # 若形状不匹配则回退为不使用 prompt（保持兼容）
            pass
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
