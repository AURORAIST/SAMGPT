import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *

# ------------------------------------------------------------
# downprompt.py：下游 few‑shot 任务使用的 prompt + 分类头。
#
# 支持 Node‑Wise Multi‑Domain Structure‑Conditioned Prompting
# ------------------------------------------------------------

def degree_structure_feature(adj):
    """与 preprompt 中完全一致的度特征提取（使用原始邻接）"""
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
    else:
        deg = adj.sum(dim=1)
    deg = torch.log1p(deg.float())
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)
    return deg.view(-1, 1)


class TensorPrompt(nn.Module):
    """把一个动态生成的 prompt 张量包装成可调用模块。"""

    def __init__(self, prompt_tensor, type_='mul'):
        super().__init__()
        self.prompt_tensor = prompt_tensor
        self.type = type_

    def forward(self, graph_embedding):
        prompt = self.prompt_tensor
        if self.type == 'add':
            if prompt.shape[0] == 1:
                prompt = prompt.repeat(graph_embedding.shape[0], 1)
            return graph_embedding + prompt
        if prompt.shape[0] == 1 and graph_embedding.shape[0] != 1:
            prompt = prompt.repeat(graph_embedding.shape[0], 1)
        return graph_embedding * prompt


class NodeWiseDomainRouter(nn.Module):
    """为每一层生成节点相关的域权重 λ_{v,d}^l"""
    def __init__(self, struct_feat_dim, num_domains, num_layers, hidden=16, tau=1.5):
        super().__init__()
        self.num_domains = num_domains
        self.tau = tau
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(struct_feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_domains)
            ) for _ in range(num_layers)
        ])
        # 初始化最后一层权重和偏置为零，使初始输出接近均匀
        for router in self.routers:
            nn.init.zeros_(router[-1].weight)
            nn.init.zeros_(router[-1].bias)

    def forward(self, S):
        # S: [N, F]
        lambda_list = []
        for router in self.routers:
            raw = router(S)                      # [N, D]
            lam = F.softmax(raw / self.tau, dim=-1)
            lambda_list.append(lam)
        return lambda_list                       # 列表，每层一个 [N, D]


class downstreamprompt(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_layers_num,
        fea_pretext_weights,
        str_pretext_weights,
        combines,
        type_='mul',
        ablation='all',
        # SCP 参数
        use_structure_conditioned_prompt=False,
        struct_feat_dim=1,
        struct_prompt_residual=True,
        struct_prompt_scale=0.05,
        # 多域参数
        num_pretrain_datasets=6,
        structure_prompt_generators=None,
    ):
        super(downstreamprompt, self).__init__()

        # ---------- 1) composed prompts ----------
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList(
            [composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
             for i in range(num_layers_num)]
        )

        # ---------- 2) open prompts ----------
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            self.open_prompt_str.append(textprompt(in_features, type_))

        # ---------- 3) 融合系数 ----------
        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation

        # ---------- 4) 结构条件 prompt 相关 ----------
        self.use_structure_conditioned_prompt = use_structure_conditioned_prompt
        self.struct_feat_dim = struct_feat_dim
        self.struct_prompt_residual = struct_prompt_residual
        self.num_pretrain_datasets = num_pretrain_datasets
        self.structure_prompt_generators = structure_prompt_generators

        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            # 可学习的全局 scale（防止 global base 被淹没）
            self.scale = nn.Parameter(torch.tensor(struct_prompt_scale))
            # node-wise domain router
            self.router = NodeWiseDomainRouter(
                struct_feat_dim=struct_feat_dim,
                num_domains=num_pretrain_datasets,
                num_layers=num_layers_num,
                hidden=16, tau=1.5
            )

    def forward(self, seq, gcn, adj, sparse, raw_adj=None):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)

        # ---------- fea 分支 ----------
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)

        if self.ablation_choice[-2:] == 'fo':
            seq_fea = self.open_prompt_fea(seq)
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = self.composedprompt_fea(seq)

        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea

        # ---------- str 分支 ----------
        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            if raw_adj is None:
                raise ValueError("raw_adj is required for structure-conditioned prompt")
            # 1) 结构特征（暂时仅支持度特征，可扩展为三维）
            S = degree_structure_feature(raw_adj)   # [N, 1]
            S = S.detach()                           # 不反向传播到结构特征

            # 2) 节点域权重 [N, D] 每层
            lambda_list = self.router(S)            # 列表，每个元素 [N, D]

            node_prompts = []
            for l in range(len(self.composedprompt_str)):
                # 全局基底：composed_token + beta * open_token
                composed_token = self.composedprompt_str[l].prompt(
                    self.composedprompt_str[l].texttoken
                ).detach().squeeze(0)
                open_token = self.open_prompt_str[l].weight.squeeze(0)
                if self.beta < 0:
                    global_base = self.weighted_prompt(
                        [composed_token.unsqueeze(0), open_token.unsqueeze(0)]
                    ).squeeze(0)
                else:
                    global_base = composed_token + self.beta * open_token

                if global_base.dim() == 1:
                    global_base = global_base.unsqueeze(0)

                # 多域生成器输出一次性堆叠 [N, D, H]
                gen_outs = torch.stack(
                    [self.structure_prompt_generators[d][l](S) for d in range(self.num_pretrain_datasets)],
                    dim=1
                )

                # 加权求和得到偏移 [N, H]
                lam = lambda_list[l].unsqueeze(-1)   # [N, D, 1]
                delta = (lam * gen_outs).sum(dim=1)   # [N, H]

                # 组合节点 prompt
                if self.struct_prompt_residual:
                    node_prompt = global_base.view(1, -1) + self.scale * delta
                else:
                    node_prompt = self.scale * delta
                node_prompts.append(TensorPrompt(node_prompt, self.composedprompt_str[l].type))

            embed_str = gcn(seq, adj, sparse, None, node_prompts)

            if self.ablation_choice == 'st':
                return embed_str
        else:
            # 原始分支（无 SCP）
            composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
            open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)
            if self.beta < 0:
                embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
            else:
                embed_str = composed_embed_str + self.beta * open_embed_str
            if self.ablation_choice[:2] == 'so':
                embed_str = open_embed_str
            elif self.ablation_choice[:2] == 'sc':
                embed_str = composed_embed_str
            if self.ablation_choice == 'st':
                return embed_str

        # 最终融合
        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    def __init__(
        self,
        ft_in,
        nb_classes,
        feature_dim,
        num_layers_num,
        fea_pretext_weights,
        str_pretext_weights,
        combines,
        type_='mul',
        ablation='all',
        use_structure_conditioned_prompt=False,
        struct_feat_dim=1,
        struct_prompt_residual=True,
        struct_prompt_scale=0.05,
        num_pretrain_datasets=6,
        structure_prompt_generators=None,
    ):
        super(downprompt, self).__init__()
        self.downstreamPrompt = downstreamprompt(
            feature_dim,
            ft_in,
            num_layers_num,
            fea_pretext_weights,
            str_pretext_weights,
            combines,
            type_,
            ablation,
            use_structure_conditioned_prompt=use_structure_conditioned_prompt,
            struct_feat_dim=struct_feat_dim,
            struct_prompt_residual=struct_prompt_residual,
            struct_prompt_scale=struct_prompt_scale,
            num_pretrain_datasets=num_pretrain_datasets,
            structure_prompt_generators=structure_prompt_generators,
        )
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0, raw_adj=None):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse, raw_adj=raw_adj).squeeze(0)
        rawret = embeds[idx]
        num = rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)

        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num, num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    def __init__(
        self,
        ft_in,
        nb_classes,
        feature_dim,
        num_layers_num,
        fea_pretext_weights,
        str_pretext_weights,
        combines,
        type_='mul',
        ablation='all',
        use_structure_conditioned_prompt=False,
        struct_feat_dim=1,
        struct_prompt_residual=True,
        struct_prompt_scale=0.05,
        num_pretrain_datasets=6,
        structure_prompt_generators=None,
    ):
        super(downprompt_graph, self).__init__()
        self.downstreamPrompt = downstreamprompt(
            feature_dim,
            ft_in,
            num_layers_num,
            fea_pretext_weights,
            str_pretext_weights,
            combines,
            type_,
            ablation,
            use_structure_conditioned_prompt=use_structure_conditioned_prompt,
            struct_feat_dim=struct_feat_dim,
            struct_prompt_residual=struct_prompt_residual,
            struct_prompt_scale=struct_prompt_scale,
            num_pretrain_datasets=num_pretrain_datasets,
            structure_prompt_generators=structure_prompt_generators,
        )
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, batch, labels=None, train=0, raw_adj=None):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse, raw_adj=raw_adj).squeeze(0)
        rawret = torch_scatter.scatter(src=embeds[idx], index=batch, dim=0, reduce='mean')
        num = rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)

        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num, num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def averageemb(labels, rawret):
    return torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')