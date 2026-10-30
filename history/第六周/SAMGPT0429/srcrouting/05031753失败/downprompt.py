import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *

# ------------------------------------------------------------
# 封装节点提示张量为可调用模块
# ------------------------------------------------------------
class NodeWisePrompt(nn.Module):
    def __init__(self, prompt_tensor):
        super().__init__()
        self.register_buffer('prompt', prompt_tensor)

    def forward(self, graph_output):
        if graph_output.dim() == 3 and graph_output.size(0) == 1:
            graph_output = graph_output.squeeze(0)
            graph_output = graph_output * self.prompt
            graph_output = graph_output.unsqueeze(0)
        else:
            graph_output = graph_output * self.prompt
        return graph_output

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


class NodeWiseDomainRouter(nn.Module):
    def __init__(self, struct_feat_dim, num_domains, num_layers, hidden=16, tau=1.5):
        super().__init__()
        self.tau = tau
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(struct_feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_domains)
            ) for _ in range(num_layers)
        ])
        for router in self.routers:
            nn.init.zeros_(router[-1].weight)
            nn.init.zeros_(router[-1].bias)

    def forward(self, S):
        lam_list = []
        for router in self.routers:
            raw = router(S)
            lam = F.softmax(raw / self.tau, dim=-1)
            lam_list.append(lam)
        return lam_list


class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all',
                 use_structure_conditioned_prompt=False,
                 struct_feat_dim=1,
                 struct_prompt_residual=True,
                 struct_prompt_scale=0.05,
                 num_pretrain_datasets=6,
                 structure_prompt_generators=None,
                 freeze_generators=True,
                 router_tau=1.5):
        super(downstreamprompt, self).__init__()
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList(
            [composedtoken([pretext[i] for pretext in str_pretext_weights], type_) for i in range(num_layers_num)]
        )
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            self.open_prompt_str.append(textprompt(in_features, type_))

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation

        self.use_structure_conditioned_prompt = use_structure_conditioned_prompt
        self.struct_feat_dim = struct_feat_dim
        self.struct_prompt_residual = struct_prompt_residual
        self.num_pretrain_datasets = num_pretrain_datasets
        self.str_pretext_weights = str_pretext_weights
        self.structure_prompt_generators = structure_prompt_generators

        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            self.scale = struct_prompt_scale
            self.router = NodeWiseDomainRouter(
                struct_feat_dim=struct_feat_dim,
                num_domains=num_pretrain_datasets,
                num_layers=num_layers_num,
                hidden=16,
                tau=router_tau
            )
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

            if self.struct_feat_dim == 1:
                S = degree_structure_feature(raw_adj)
            elif self.struct_feat_dim == 3:
                S = three_feat_structure_feature(raw_adj)
            else:
                raise ValueError("struct_feat_dim must be 1 or 3")
            S = S.detach()

            lambda_list = self.router(S)
            self.last_lambda_list = lambda_list

            if step is not None and step % 50 == 0:
                with torch.no_grad():
                    print(f"Step {step}: λ[0] mean = {lambda_list[0].mean(dim=0).detach().cpu().numpy()}")
                    print(f"Step {step}: scale = {self.scale:.4f}")

            composed_prompts = []
            for l in range(len(self.str_pretext_weights[0])):
                pre_tokens = torch.stack(
                    [self.str_pretext_weights[d][l].squeeze(0) for d in range(self.num_pretrain_datasets)],
                    dim=0
                )  # [D, H]
                lam = lambda_list[l]  # [N, D]
                composed_token_v = torch.matmul(lam, pre_tokens)  # [N, H]

                open_token = self.open_prompt_str[l].weight.squeeze(0)  # [H]
                if self.beta < 0:
                    global_base = composed_token_v + open_token.unsqueeze(0)
                else:
                    global_base = composed_token_v + self.beta * open_token.unsqueeze(0)

                gen_outs = torch.stack(
                    [self.structure_prompt_generators[d][l](S) for d in range(self.num_pretrain_datasets)],
                    dim=1
                )  # [N, D, H]
                delta = (lam.unsqueeze(-1) * gen_outs).sum(dim=1)  # [N, H]

                if self.struct_prompt_residual:
                    node_prompt = global_base + self.scale * delta
                else:
                    node_prompt = self.scale * delta

                # 包装为可调用模块
                composed_prompts.append(NodeWisePrompt(node_prompt))

            # 双流注入：composed 流使用包装后的模块，open 流保持不变
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
        else:
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
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all',
                 use_structure_conditioned_prompt=False,
                 struct_feat_dim=1,
                 struct_prompt_residual=True,
                 struct_prompt_scale=0.05,
                 num_pretrain_datasets=6,
                 structure_prompt_generators=None,
                 freeze_generators=True,
                 router_tau=1.5):
        super(downprompt, self).__init__()
        self.downstreamPrompt = downstreamprompt(
            feature_dim, ft_in, num_layers_num,
            fea_pretext_weights, str_pretext_weights, combines,
            type_, ablation,
            use_structure_conditioned_prompt=use_structure_conditioned_prompt,
            struct_feat_dim=struct_feat_dim,
            struct_prompt_residual=struct_prompt_residual,
            struct_prompt_scale=struct_prompt_scale,
            num_pretrain_datasets=num_pretrain_datasets,
            structure_prompt_generators=structure_prompt_generators,
            freeze_generators=freeze_generators,
            router_tau=router_tau,
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
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all',
                 use_structure_conditioned_prompt=False,
                 struct_feat_dim=1,
                 struct_prompt_residual=True,
                 struct_prompt_scale=0.05,
                 num_pretrain_datasets=6,
                 structure_prompt_generators=None,
                 freeze_generators=True,
                 router_tau=1.5):
        super(downprompt_graph, self).__init__()
        self.downstreamPrompt = downstreamprompt(
            feature_dim, ft_in, num_layers_num,
            fea_pretext_weights, str_pretext_weights, combines,
            type_, ablation,
            use_structure_conditioned_prompt=use_structure_conditioned_prompt,
            struct_feat_dim=struct_feat_dim,
            struct_prompt_residual=struct_prompt_residual,
            struct_prompt_scale=struct_prompt_scale,
            num_pretrain_datasets=num_pretrain_datasets,
            structure_prompt_generators=structure_prompt_generators,
            freeze_generators=freeze_generators,
            router_tau=router_tau,
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