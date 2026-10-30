import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *

# ------------------------------------------------------------
# downprompt.py：下游 few-shot 任务使用的 prompt + 分类头。
#
# 核心思想：
# - 预训练阶段得到一组“预训练 prompt 权重”（fea_pretext_weights / str_pretext_weights）
# - 下游阶段再引入“open prompt”（可训练）
# - 将 composed prompt（由多个预训练 prompt 融合）和 open prompt 组合，生成下游 embedding
# - 支持结构条件 prompt（node-conditioned）：使用预训练的 structure_prompt_generators
#   对下游图计算 degree-based 节点 prompt，注入 GCN，保持与预训练一致。
# - 分类采用“类原型/均值向量 + 余弦相似度”的非参数方式。
# ------------------------------------------------------------

def degree_structure_feature(adj):
    """从邻接矩阵构建节点度特征（与 preprompt 中完全一致）。

    Args:
        adj: torch Tensor, 2D dense or sparse, 形状 [N, N]

    Returns:
        S: [N, 1] 标准化后的 log-degree
    """
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
    else:
        deg = adj.sum(dim=1)
    deg = torch.log1p(deg.float())
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)
    return deg.view(-1, 1)


class downstreamprompt(nn.Module):
    """下游 prompt 组合模块，支持结构条件 prompt。

    参数说明：
    - feature_dim: 输入特征维度
    - hidden_dim: 隐层维度（GCN 输出维度）
    - num_layers_num: GNN 层数
    - fea_pretext_weights: 来自预训练的特征 prompt 权重列表
    - str_pretext_weights: 来自预训练的结构 prompt 权重列表（每个元素是 list of Tensor [1,dim]）
    - combines: 融合系数 [alpha, beta?]；alpha 用于 fea/str 融合，beta 用于 composed/open 融合
    - type_: prompt 类型 'add'/'mul'
    - ablation: 消融选项
    - use_structure_conditioned_prompt: 是否启用节点结构条件 prompt
    - struct_feat_dim: 结构特征维度
    - struct_prompt_residual: 是否使用残差形式 global + node_delta
    - structure_prompt_generators: 预训练的节点 prompt 生成器列表（每个元素是每层的 MLP）
    - pretrain_dataset_idx: 从 str_pretext_weights 选择哪个预训练数据集的全局 prompt
    """

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
        use_structure_conditioned_prompt=False,
        struct_feat_dim=1,
        struct_prompt_residual=True,
        structure_prompt_generators=None,
        pretrain_dataset_idx=0,
    ):
        super(downstreamprompt, self).__init__()

        # -------- 1) composed prompts（融合预训练 prompt）--------
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)

        self.composedprompt_str = nn.ModuleList(
            [
                composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
                for i in range(num_layers_num)
            ]
        )

        # -------- 2) open prompts（下游可训练 prompt）--------
        self.open_prompt_fea = textprompt(feature_dim)

        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:          # 使用第一个数据集的权重形状
            in_features = weight.size(1)
            new_layer = textprompt(in_features, type_)
            self.open_prompt_str.append(new_layer)

        # -------- 3) 融合系数 --------
        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)

        # -------- 4) 结构条件 prompt 相关 --------
        self.use_structure_conditioned_prompt = use_structure_conditioned_prompt
        self.struct_feat_dim = struct_feat_dim
        self.struct_prompt_residual = struct_prompt_residual
        self.structure_prompt_generators = structure_prompt_generators
        self.ablation_choice = ablation

        # 提取全局 prompt token（每层），用于节点条件 prompt 的残差基底
        self.global_tokens = []
        for layer_idx in range(num_layers_num):
            # 使用 pretrain_dataset_idx 指定预训练数据集
            weight = str_pretext_weights[pretrain_dataset_idx][layer_idx]
            if weight.dim() == 2:
                weight = weight.squeeze(0)   # [hidden_dim]
            self.global_tokens.append(weight)

    def forward(self, seq, gcn, adj, sparse):
        """生成下游 embedding。

        Args:
            seq: 输入节点特征 [N,F] 或 [1,N,F]
            gcn: 预训练的 GNN 主干
            adj: 邻接矩阵
            sparse: 是否稀疏

        Returns:
            ret: 节点 embedding
        """
        # 特殊消融：完全不用 prompt
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)

        # ---------- fea prompt 分支 ----------
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)

        # fea-only 消融
        if self.ablation_choice[-2:] == 'fo':
            seq_fea = self.open_prompt_fea(seq)
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = self.composedprompt_fea(seq)

        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea

        # ---------- str prompt 分支 ----------
        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            # 构建节点级 prompt（与预训练完全一致）
            S = degree_structure_feature(adj)   # [N, 1]
            node_prompts = []
            for layer_idx in range(len(self.global_tokens)):
                generator = self.structure_prompt_generators[0][layer_idx]  # 使用第一个预训练数据集的生成器
                node_delta = generator(S)
                node_delta = torch.tanh(node_delta)                                     # [N, hidden_dim]
                global_token = self.global_tokens[layer_idx]
                if self.struct_prompt_residual:
                    node_prompt = global_token.unsqueeze(0) + 0.00 * node_delta   # scale 改为 0.05
                else:
                    node_prompt = 0.00 * node_delta           
                node_prompts.append(node_prompt)
            embed_str = gcn(seq, adj, sparse, None, node_prompts)

            # 消融：str-only 返回
            if self.ablation_choice == 'st':
                return embed_str
        else:
            # 原始分支：使用 composed/open prompt 模块
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

        # ---------- 最终融合：fea + alpha * str ----------
        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    """节点级 few-shot 分类头。"""

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
        structure_prompt_generators=None,
        pretrain_dataset_idx=0,
    ):
        super(downprompt, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)

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
            structure_prompt_generators=structure_prompt_generators,
            pretrain_dataset_idx=pretrain_dataset_idx,
        )

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)
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
    """图级 few-shot 分类头。"""

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
        structure_prompt_generators=None,
        pretrain_dataset_idx=0,
    ):
        super(downprompt_graph, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)

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
            structure_prompt_generators=structure_prompt_generators,
            pretrain_dataset_idx=pretrain_dataset_idx,
        )

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, batch, labels=None, train=0):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)
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
    """按 labels 对 rawret 做 mean 聚合，得到每个类别的原型向量。

    Args:
        labels: Tensor[num]
        rawret: Tensor[num, F]

    Returns:
        Tensor[C, F]
    """
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel