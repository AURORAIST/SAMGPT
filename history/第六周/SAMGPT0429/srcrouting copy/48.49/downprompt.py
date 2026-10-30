import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *

# ------------------------------------------------------------
# 结构特征函数（与 preprompt 完全一致）
# ------------------------------------------------------------
def degree_structure_feature(adj):
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
    else:
        deg = adj.sum(dim=1)
    deg = torch.log1p(deg.float())
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)
    return deg.view(-1, 1)

def three_feat_structure_feature(adj):
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        adj_dense = adj.to_dense()
    else:
        deg = adj.sum(dim=1)
        adj_dense = adj
    deg = torch.log1p(deg.float())

    tri = torch.diag(torch.matrix_power(adj_dense.float(), 3)) / 2.0
    deg_clamp = deg.float()
    clustering = 2.0 * tri / (deg_clamp * (deg_clamp - 1) + 1e-8)
    clustering[deg_clamp <= 1] = 0.0

    adj_cpu = adj_dense.cpu().float()
    N = adj_cpu.shape[0]
    deg_sum_cpu = adj_cpu.sum(dim=1).clamp(min=1e-6)
    adj_norm_cpu = adj_cpu / deg_sum_cpu.unsqueeze(1)
    pr_cpu = torch.ones(N) / N
    for _ in range(3):
        pr_cpu = 0.85 * torch.mv(adj_norm_cpu.T, pr_cpu) + 0.15 / N
    pr = torch.log1p(pr_cpu * N).to(adj_dense.device)

    def norm(x): return (x - x.mean()) / (x.std() + 1e-6)
    deg = norm(deg)
    clustering = norm(clustering)
    pr = norm(pr)
    return torch.stack([deg, clustering, pr], dim=1)


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
        num_pretrain_datasets=6,
        structure_prompt_generators=None,
        freeze_generators=True,
    ):
        super(downstreamprompt, self).__init__()

        # -------- 1) composed/open prompts（保留原始模块，用于无 SCP 分支）--------
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList(
            [composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
             for i in range(num_layers_num)]
        )

        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            self.open_prompt_str.append(textprompt(in_features, type_))

        # -------- 2) 融合系数 --------
        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation

        # -------- 3) SCP 相关 --------
        self.use_structure_conditioned_prompt = use_structure_conditioned_prompt
        self.struct_feat_dim = struct_feat_dim
        self.struct_prompt_residual = struct_prompt_residual
        self.struct_prompt_scale = struct_prompt_scale
        self.num_pretrain_datasets = num_pretrain_datasets
        self.structure_prompt_generators = structure_prompt_generators

        # 保存预训练权重以便在 SCP 分支中自行融合
        self.str_pretext_weights = str_pretext_weights

        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            self.scale = nn.Parameter(torch.tensor(struct_prompt_scale))
            # 每层一个全局域权重向量（初始均匀）
            self.domain_weights = nn.ParameterList([
                nn.Parameter(torch.zeros(num_pretrain_datasets))
                for _ in range(num_layers_num)
            ])

            # 控制生成器是否冻结
            if freeze_generators:
                for gen_list in self.structure_prompt_generators:
                    for gen in gen_list:
                        for p in gen.parameters():
                            p.requires_grad = False

    def forward(self, seq, gcn, adj, sparse, raw_adj=None, step=None):
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

            # 1. 结构特征
            if self.struct_feat_dim == 1:
                S = degree_structure_feature(raw_adj)
            elif self.struct_feat_dim == 3:
                S = three_feat_structure_feature(raw_adj)
            else:
                raise ValueError("struct_feat_dim must be 1 or 3")
            S = S.detach()

            # 2. 为每一层构造 composed prompt
            composed_prompts = []
            for l in range(self.num_layers):
                # 域权重 (全局 lambda)
                lam = F.softmax(self.domain_weights[l], dim=0)  # [D]

                # 全局基底：加权平均预训练权重 + open token
                pre_weights = torch.stack(
                    [self.str_pretext_weights[d][l].squeeze(0) for d in range(self.num_pretrain_datasets)],
                    dim=0
                )  # [D, H]
                composed_token = (lam.unsqueeze(1) * pre_weights).sum(dim=0)  # [H]

                open_token = self.open_prompt_str[l].weight.squeeze(0)
                if self.beta < 0:
                    global_base = self.weighted_prompt(
                        [composed_token.unsqueeze(0), open_token.unsqueeze(0)]
                    ).squeeze(0)
                else:
                    global_base = composed_token + self.beta * open_token

                # 多域生成器融合偏移
                gen_outs = torch.stack(
                    [self.structure_prompt_generators[d][l](S) for d in range(self.num_pretrain_datasets)],
                    dim=1
                )  # [N, D, H]
                delta = (lam.view(1, -1, 1) * gen_outs).sum(dim=1)  # [N, H]

                # 节点提示
                if self.struct_prompt_residual:
                    node_prompt = global_base.view(1, -1) + self.scale * delta
                else:
                    node_prompt = self.scale * delta
                composed_prompts.append(node_prompt)

            # 3. 分别注入 GCN（双流）
            composed_embed_str = gcn(seq, adj, sparse, None, composed_prompts)
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

            # 调试打印
            if step is not None and step % 50 == 0:
                print(f"DEBUG domain lambda (layer0): {lam.detach().cpu().numpy()}")
                print(f"DEBUG scale: {self.scale.item():.4f}")

        else:
            # 原始无 SCP 分支（完全保留原始逻辑）
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
        freeze_generators=True,
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
            freeze_generators=freeze_generators,
        )
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0, raw_adj=None, step=None):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse, raw_adj=raw_adj, step=step).squeeze(0)
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
        freeze_generators=True,
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
            freeze_generators=freeze_generators,
        )
        self.nb_classes = nb_classes
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def forward(self, features, adj, sparse, gcn, idx, batch, labels=None, train=0, raw_adj=None, step=None):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse, raw_adj=raw_adj, step=step).squeeze(0)
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