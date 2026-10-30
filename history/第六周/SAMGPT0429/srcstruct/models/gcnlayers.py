import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN
from layers.prompt import *


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super(GcnLayers, self).__init__()

        self.act = torch.nn.ReLU()
        self.num_layers_num = num_layers_num
        self.g_net, self.bns = self.create_net(n_in, n_h, self.num_layers_num)
        self.dropout = torch.nn.Dropout(p=dropout)

    def create_net(self, input_dim, hidden_dim, num_layers):
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            if i:
                gcn_layer = GCN(hidden_dim, hidden_dim)
            else:
                gcn_layer = GCN(input_dim, hidden_dim)

            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(gcn_layer)
            self.bns.append(bn)

        return self.convs, self.bns

    def _apply_prompt(self, graph_output, prompt):
        """
        支持两种 prompt：

        1. 原始 SAMGPT:
           prompt 是 textprompt module
           用法: prompt(graph_output)

        2. 新的 structure-conditioned prompt:
           prompt 是 Tensor, shape = [num_nodes, hidden_dim]
           用法: graph_output * prompt
        """
        if torch.is_tensor(prompt):
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)

            prompt = prompt.to(
                device=graph_output.device,
                dtype=graph_output.dtype
            )

            if prompt.shape[0] == 1:
                return graph_output * prompt

            if prompt.shape[0] != graph_output.shape[0]:
                raise ValueError(
                    f"Node prompt size mismatch: "
                    f"prompt has {prompt.shape[0]} nodes, "
                    f"but graph_output has {graph_output.shape[0]} nodes."
                )

            return graph_output * prompt

        else:
            return prompt(graph_output)

    def forward(self, seq, adj, sparse, LP=False, prompt_layers=None):
        graph_output = torch.squeeze(seq, dim=0)

        if prompt_layers is not None:
            assert len(prompt_layers) == self.num_layers_num

        xs = []

        for i in range(self.num_layers_num):
            input_data = (graph_output, adj)

            if i:
                graph_output = self.convs[i](input_data) + graph_output
            else:
                graph_output = self.convs[i](input_data)

            if prompt_layers is not None:
                graph_output = self._apply_prompt(
                    graph_output,
                    prompt_layers[i]
                )

            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)

            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0)
