import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *
from layers.archetype import DomainAwareAA


class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all'):
        super(downstreamprompt, self).__init__()
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList([
            composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
            for i in range(num_layers_num)
        ])
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            self.open_prompt_str.append(textprompt(in_features, type_))
        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)

        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)

        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)

        if self.ablation_choice[-2:] == 'fo':
            seq_fea == open_seq_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea

        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea

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

        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    """节点级 few-shot 分类头，支持 DA-AA 下游原型增强。

    DA-AA 在目标域全图嵌入上训练原型（而非预训练域），
    解决预训练域原型不适用于目标域的域不匹配问题。
    """

    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all'):
        super(downprompt, self).__init__()
        self.num_pretrain_datasets = len(fea_pretext_weights)
        self.downstreamPrompt = downstreamprompt(
            feature_dim, ft_in, num_layers_num,
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.zeros(nb_classes, ft_in)  # 初始化为零
        self._ft_in = ft_in

        # DA-AA 下游原型增强（默认关闭，通过 enable_daaa 启用）
        self.use_daaa = False
        self._daaa_initialized = False
        self._daaa_loss = torch.tensor(0.0)

    def enable_daaa(self, n_shared=4, n_domain=2, lambda_vert=0.01, lambda_res=0.1):
        """启用下游 DA-AA 原型增强。
        
        n_domain=0 表示仅使用共享原型（用于 CDG 对比实验）。
        """
        self.daaa = DomainAwareAA(
            n_domains=1,                # 下游只涉及目标域一个域
            n_shared=n_shared,
            n_domain=max(n_domain, 1),  # 至少1个，避免空列表
            dim=self._ft_in,
            lambda_vert=lambda_vert,
            lambda_res=lambda_res,
        )
        self._daaa_shared_only = (n_domain == 0)
        self.use_daaa = True
        self._daaa_initialized = False

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)

        # DA-AA 增强：用目标域全图嵌入训练原型并增强
        if self.use_daaa:
            if not self._daaa_initialized:
                # 用目标域初始嵌入初始化原型（取极端点）
                self.daaa.init_from_data([embeds.detach().cpu()])
                self.daaa = self.daaa.to(embeds.device)
                self._daaa_initialized = True

            # shared_only 模式：仅使用共享原型（消融对比用）
            if getattr(self, '_daaa_shared_only', False):
                with torch.no_grad() if train == 0 else torch.enable_grad():
                    z_s = self.daaa.z_shared.to(embeds.device)
                    a_s = self.daaa._compute_coefficients(embeds.detach(), z_s)
                    h_out = embeds.detach() + 0.3 * (a_s @ z_s - embeds.detach())
                self._daaa_loss = torch.tensor(0.0, device=embeds.device)
            elif train == 1:
                # 训练时：原型参数也参与更新（提供辅助损失）
                h_out, aa_losses = self.daaa(embeds.detach(), domain_id=0)
                self._daaa_loss = aa_losses['total']
            else:
                # 评估时：仅前向，不更新原型
                with torch.no_grad():
                    h_out, _ = self.daaa(embeds.detach(), domain_id=0)
            embeds = h_out

        rawret = embeds[idx]
        num = rawret.shape[0]

        if train == 1:
            # 计算类别 prototype
            self.ave = averageemb(labels=labels, rawret=rawret)

        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num, num:]
        ret = F.softmax(ret, dim=1)
        return ret


    def get_daaa_loss(self):
        """获取 DA-AA 辅助损失，用于下游训练时加到总损失中。"""
        if self.use_daaa:
            return self._daaa_loss
        return torch.tensor(0.0)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all'):
        super(downprompt_graph, self).__init__()
        self.num_pretrain_datasets = len(fea_pretext_weights)
        self.downstreamPrompt = downstreamprompt(
            feature_dim, ft_in, num_layers_num,
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.zeros(nb_classes, ft_in)  # 初始化为零，防止未训练时用随机内存推理

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
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel
