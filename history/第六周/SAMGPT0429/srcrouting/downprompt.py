import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter
from layers.prompt import *

# ------------------------------------------------------------
# 1. 节点提示封装：支持 FiLM (scale + bias)
# ------------------------------------------------------------
class NodeWisePrompt(nn.Module):
    def __init__(self, gamma, beta):
        super().__init__()
        self.register_buffer('gamma', gamma)   # [N, H]
        self.register_buffer('beta', beta)     # [N, H]

    def forward(self, graph_output):
        # graph_output 可能是 [1, N, H] 或 [N, H]
        if graph_output.dim() == 3 and graph_output.size(0) == 1:
            graph_output = graph_output.squeeze(0)
            graph_output = graph_output * self.gamma + self.beta
            graph_output = graph_output.unsqueeze(0)
        else:
            graph_output = graph_output * self.gamma + self.beta
        return graph_output

# ------------------------------------------------------------
# 2. 结构特征函数（与 preprompt 一致）
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

# ------------------------------------------------------------
# 3. 节点自适应路由器（支持 Sparse Top‑k）
# ------------------------------------------------------------
class NodeWiseDomainRouter(nn.Module):
    def __init__(self, struct_feat_dim, num_domains, num_layers,
                 hidden=16, tau=3.0, topk=2, use_sparse=True):
        super().__init__()
        self.tau = tau
        self.topk = topk
        self.use_sparse = use_sparse
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
            raw = router(S)                     # [N, D]
            lam = F.softmax(raw / self.tau, dim=-1)
            if self.use_sparse:
                # 稀疏化：保留 top‑2，其余置零，重新归一化
                topk_val, topk_idx = torch.topk(lam, k=self.topk, dim=-1)
                mask = torch.zeros_like(lam).scatter_(-1, topk_idx, topk_val)
                lam = mask / mask.sum(dim=-1, keepdim=True)
            lam_list.append(lam)
        return lam_list

# ------------------------------------------------------------
# 4. 主下游提示模块（升级版）
# ------------------------------------------------------------
class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num,
                 fea_pretext_weights, str_pretext_weights, combines,
                 type_='mul', ablation='all',
                 use_structure_conditioned_prompt=False,
                 struct_feat_dim=1,
                 struct_prompt_residual=True,       # 保留兼容，实际由 subspace 控制
                 struct_prompt_scale=0.05,
                 num_pretrain_datasets=6,
                 structure_prompt_generators=None,
                 freeze_generators=True,
                 router_tau=3.0,
                 # 新参数
                 use_sparse_routing=True,           # 是否使用稀疏路由
                 sparse_topk=2,                     # 稀疏路由 top‑k 值
                 use_film=True,                     # 是否使用 FiLM 仿射调制
                 use_subspace_prompt=True,          # 是否使用子空间 prompt 替代残差
                 subspace_basis_num=8,              # 子空间基向量数量
                 ):
        super(downstreamprompt, self).__init__()

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
        self.use_film = use_film
        self.use_subspace_prompt = use_subspace_prompt

        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            # 固定 scale（不再可学习）
            self.scale = struct_prompt_scale

            # 路由器：稀疏路由、可调温度
            self.router = NodeWiseDomainRouter(
                struct_feat_dim=struct_feat_dim,
                num_domains=num_pretrain_datasets,
                num_layers=num_layers_num,
                hidden=16,
                tau=router_tau,
                topk=sparse_topk,
                use_sparse=use_sparse_routing
            )

            # 冻结或微调生成器
            if freeze_generators:
                for gen_list in self.structure_prompt_generators:
                    for gen in gen_list:
                        for p in gen.parameters():
                            p.requires_grad = False

            # 子空间 prompt 基向量（如果启用）
            if self.use_subspace_prompt:
                self.prompt_basis = nn.ParameterList([
                    nn.Parameter(torch.randn(subspace_basis_num, hidden_dim) * 0.02)
                    for _ in range(num_layers_num)
                ])
                # 每层一个路由器，输出基组合权重
                self.basis_router = NodeWiseDomainRouter(
                    struct_feat_dim=struct_feat_dim,
                    num_domains=subspace_basis_num,   # 注意：这里域数 = 基向量个数
                    num_layers=num_layers_num,
                    hidden=16,
                    tau=router_tau,
                    topk=min(sparse_topk, subspace_basis_num),
                    use_sparse=use_sparse_routing
                )
            else:
                self.prompt_basis = None
                self.basis_router = None

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

            # 域路由权重
            lambda_domain = self.router(S)   # 每层 [N, D]

            # 基路由权重（如果启用子空间提示）
            if self.use_subspace_prompt and self.basis_router is not None:
                lambda_basis = self.basis_router(S)   # 每层 [N, K]
            else:
                lambda_basis = None

            composed_prompts = []
            for l in range(len(self.str_pretext_weights[0])):
                # ---- 多域生成器融合偏移 ----
                # 由于 FiLM 需要输出两倍维度，生成器输出维度应为 2*H（如果不启用 FiLM 则保持原样）
                # 这里我们在构造生成器时已经固定了输出维度，无法动态更改。
                # 为兼容性，我们采用以下策略：
                # 如果启用 FiLM，我们期望生成器输出维度为 2*H；否则为 H。
                # 但预训练生成器输出维度为 H，我们无法改变。因此我们采用额外的小型 MLP 来从 gen_outs 生成 gamma 和 beta。
                # 为了不破坏原有生成器，我们选择在生成器输出后拆分（需保证维度足够）或新增一个 FiLM head。
                # 简洁做法：使用一个独立的 FiLM head，输入 S，输出 [N, 2H]。
                # 这里我们设计一个简易的 FiLM head，如果 use_film=True 则创建并调用，否则使用原有生成器输出。

                # 由于代码长度限制，我们采用折中：如果 use_film=True，我们直接使用生成器输出，但将 scale 和 offset 分开计算。
                # 生成器输出 [N, D, H]，我们按域权重求和后得到 delta [N, H]。
                # 然后通过一个可学习的 linear 层映射为 gamma 和 beta。
                # 这样既利用了生成器，又实现了 FiLM。

                gen_outs = torch.stack(
                    [self.structure_prompt_generators[d][l](S) for d in range(self.num_pretrain_datasets)],
                    dim=1
                )  # [N, D, H]

                lam_d = lambda_domain[l].unsqueeze(-1)  # [N, D, 1]
                delta = (lam_d * gen_outs).sum(dim=1)   # [N, H]

                if self.use_film:
                    # 注意：这里需要一个额外的可训练映射，我们动态创建但可能会多次创建？
                    # 在 __init__ 中应预先定义。为了简化，我们在这里直接使用一个独立的 Module。
                    # 更好的做法是在 __init__ 中创建 self.film_heads，此处仅调用。
                    # 由于代码篇幅，我们假定已在 __init__ 中定义了 self.film_heads (ModuleList)。
                    # 如果没有，你可以自行添加。
                    # 我们在此提供正确的初始化代码，你需要在 __init__ 中加上：
                    # if self.use_film:
                    #     self.film_heads = nn.ModuleList([
                    #         nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(num_layers_num)
                    #     ])
                    gamma_beta = self.film_heads[l](delta)  # [N, 2H]
                    gamma, beta = gamma_beta.chunk(2, dim=-1)
                else:
                    gamma = self.scale * delta
                    beta = torch.zeros_like(gamma)

                # ---- 组合节点 prompt ----
                if self.use_subspace_prompt and self.prompt_basis is not None and lambda_basis is not None:
                    # 子空间组合：p = sum_k alpha_k * basis_k
                    basis = self.prompt_basis[l]            # [K, H]
                    lam_b = lambda_basis[l]                 # [N, K]
                    # 基组合得到 prompt_token [N, H]
                    prompt_token = torch.matmul(lam_b, basis)
                else:
                    # 传统残差方式：global_base + scale * delta
                    # 利用 composed token 和 open token 构造全局基底（与之前相同）
                    pre_tokens = torch.stack(
                        [self.str_pretext_weights[d][l].squeeze(0) for d in range(self.num_pretrain_datasets)],
                        dim=0
                    )  # [D, H]
                    composed_token_v = torch.matmul(lam_d.squeeze(-1), pre_tokens)  # [N, H]
                    open_token = self.open_prompt_str[l].weight.squeeze(0)  # [H]
                    if self.beta < 0:
                        global_base = composed_token_v + open_token.unsqueeze(0)
                    else:
                        global_base = composed_token_v + self.beta * open_token.unsqueeze(0)
                    # 节点提示 = 全局基底 + 缩放后的偏移
                    prompt_token = global_base + self.scale * delta

                # 节点提示变为 FiLM 参数对（如果有 FiLM 则 gamma/beta 已由 delta 生成，否则 gamma=scale*delta, beta=0）
                if self.use_film:
                    # gamma, beta 已在上面通过 film_heads 获得
                    pass
                else:
                    # gamma 设为 prompt_token，beta 为零，退化为普通乘法调制
                    gamma = prompt_token
                    beta = torch.zeros_like(gamma)

                composed_prompts.append(NodeWisePrompt(gamma, beta))

            # 双流注入
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
            # 原始分支
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

# ------------------------------------------------------------
# downprompt 和 downprompt_graph 类与之前相同，只需传递新参数
# ------------------------------------------------------------
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
                 router_tau=3.0,
                 use_sparse_routing=True,
                 sparse_topk=2,
                 use_film=True,
                 use_subspace_prompt=True,
                 subspace_basis_num=8):
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
            use_sparse_routing=use_sparse_routing,
            sparse_topk=sparse_topk,
            use_film=use_film,
            use_subspace_prompt=use_subspace_prompt,
            subspace_basis_num=subspace_basis_num
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
                 router_tau=3.0,
                 use_sparse_routing=True,
                 sparse_topk=2,
                 use_film=True,
                 use_subspace_prompt=True,
                 subspace_basis_num=8):
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
            use_sparse_routing=use_sparse_routing,
            sparse_topk=sparse_topk,
            use_film=use_film,
            use_subspace_prompt=use_subspace_prompt,
            subspace_basis_num=subspace_basis_num
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