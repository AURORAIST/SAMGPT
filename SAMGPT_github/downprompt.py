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
# - 分类采用“类原型/均值向量 + 余弦相似度”的非参数方式：
#   先按标签对支持集求每类平均 embedding(ave)，再用 cosine similarity 得到每个样本对各类的相似度并 softmax。
# ------------------------------------------------------------


class downstreamprompt(nn.Module):
    """下游 prompt 组合模块：给 GCN 编码前/编码中注入 prompt。

    这里区分两种 prompt：
    1) composed prompt：由多个预训练 prompt 通过 weighted_prompt/composedtoken 融合得到
    2) open prompt：下游额外引入的可训练 prompt（类似“可适配”的文本提示）

    同时区分两条支路：
    - fea 分支：直接对输入特征 seq 做 prompt 调制，再送入 gcn(seq_fea, ...)
    - str 分支：把 prompt 作为“结构/层级 prompt”（传给 gcn 的 prompt 参数）影响每层表示

    ablation_choice 用于控制消融实验（选择只用某部分）。
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
    ):
        super(downstreamprompt, self).__init__()

        # -------- 1) composed prompts（融合预训练 prompt）--------
        # fea_pretext_weights: 来自预训练的特征 prompt 列表
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)

        # str_pretext_weights: 每层都有一组 prompt（因此这里为每一层构造 composedtoken）
        # 形状/组织方式依赖预训练模型的 get_weights() 约定。
        self.composedprompt_str = nn.ModuleList(
            [
                composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
                for i in range(num_layers_num)
            ]
        )

        # -------- 2) open prompts（下游可训练 prompt）--------
        # 对输入特征的 open prompt
        self.open_prompt_fea = textprompt(feature_dim)

        # 对每层结构 prompt 的 open prompt：按 str_pretext_weights[0] 的各个 weight 的维度创建
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            new_layer = textprompt(in_features, type_)
            self.open_prompt_str.append(new_layer)
        # nn.ModuleList([textprompt(hidden_dim, type) for _ in range(num_layers_num)])

        # -------- 3) 融合系数 --------
        # alpha: 最终 ret = embed_fea + alpha * embed_str
        self.alpha = combines[0]
        # beta: composed 与 open 的融合系数（负值时走 weighted_prompt 分支）
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        # 当 beta<0 时，用 weighted_prompt 学习融合 composed/open 两支
        self.weighted_prompt = weighted_prompt(2)

        # 消融选项
        self.ablation_choice = ablation

    def forward(self, seq, gcn, adj, sparse):
        """生成下游 embedding。

        Args:
            seq: 输入节点特征（通常 [N,F] 或 [1,N,F]，取决于 gcn 实现）
            gcn: 主干 GNN（预训练模型中的 gcn），需要支持 prompt 参数
            adj: 邻接（torch sparse 或 dense）
            sparse: 是否使用稀疏邻接

        Returns:
            ret: 节点 embedding（或中间 embedding，取决于 ablation_choice）
        """
        # 特殊消融：完全不用 prompt，直接跑 gcn
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)

        # ---------- fea prompt 分支：对输入特征做调制 ----------
        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)

        # beta<0 时使用可学习 weighted_prompt 融合；否则按 composed + beta*open
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)

        # fea-only 消融：
        # 约定：ablation_choice 末尾 'fo' 表示只用 open fea；'fc' 表示只用 composed fea
        if self.ablation_choice[-2:] == 'fo':
            seq_fea = open_seq_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea

        # 得到 fea embedding
        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            # 只返回 fea 分支
            return embed_fea

        # ---------- str prompt 分支：以“层 prompt”的方式影响 gcn ----------
        composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
        open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)

        if self.beta < 0:
            embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
        else:
            embed_str = composed_embed_str + self.beta * open_embed_str

        # str-only 消融：
        # 约定：ablation_choice 前缀 'so' 表示只用 open str；'sc' 表示只用 composed str
        if self.ablation_choice[:2] == 'so':
            embed_str = open_embed_str
        elif self.ablation_choice[:2] == 'sc':
            embed_str = composed_embed_str

        if self.ablation_choice == 'st':
            # 只返回 str 分支
            return embed_str

        # ---------- 最终融合：fea + alpha * str ----------
        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    """节点级 few-shot 分类头。

    做法：
    1) 通过 downstreamprompt 得到所有节点 embedding
    2) 选取 idx 对应的 embedding 作为当前 batch/训练集样本 rawret
    3) 训练阶段 (train=1)：按标签对 rawret 求每类均值（作为类原型 ave）
    4) 计算 rawret 与 ave 的余弦相似度，作为 logits，并做 softmax

    这里没有显式的线性分类层，属于 prototype-based 分类。
    """

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
        )

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        # ave: 每个类别的“原型向量”（均值 embedding），形状 [C, F]
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0):
        # 生成全图 embedding，然后取出 idx 对应的样本
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)
        rawret = embeds[idx]
        num = rawret.shape[0]

        # 训练时更新类原型（支持集每类均值）
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)

        # 通过“拼接样本与类原型”一次性算 cosine similarity 矩阵
        # rawret: [num,F], ave:[C,F] -> cat: [num+C,F]
        rawret = torch.cat((rawret, self.ave), dim=0)
        # sim: [num+C, num+C]
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)

        # 取样本到各类别原型的相似度：[:num, num:]
        ret = rawret[:num, num:]
        # softmax 得到概率分布
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        # 线性层初始化（当前类里并未实际使用 Linear，但保留接口）
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    """图级 few-shot 分类头。

    与 downprompt 的区别：
    - idx 指向的是“子图节点索引列表”（由 build_subgraph 产生）
    - batch 指示每个节点属于哪个子图
    - 先用 torch_scatter.scatter(mean) 将节点 embedding 聚合成子图 embedding
    之后与 downprompt 相同：用类原型 + cosine similarity 做分类。
    """

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
        )

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, batch, labels=None, train=0):
        # 先得到节点 embedding
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)

        # 按 batch 将 idx 对应节点的 embedding 求均值，得到每个子图的表示（graph embedding）
        rawret = torch_scatter.scatter(src=embeds[idx], index=batch, dim=0, reduce='mean')
        num = rawret.shape[0]

        # 训练时更新类原型
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)

        # cosine similarity to prototypes
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
        labels: Tensor[num]，取值范围 [0, C-1]
        rawret: Tensor[num, F]

    Returns:
        Tensor[C, F]
    """
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel