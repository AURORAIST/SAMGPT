import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers
from layers import AvgReadout
import numpy as np
from sklearn.decomposition import PCA
from layers.prompt import *
from layers.archetype import DomainAwareAA
import copy

# ------------------------------------------------------------
# preprompt.py：预训练阶段的核心模型 PrePrompt。
#
# 新增：Structure-Conditioned Prompt (SCP)
#   在预训练时，structure prompt 不再是每个数据集一个全局向量，
#   而是由节点级结构特征（degree / clustering / PageRank）通过 MLP
#   生成节点级偏移，叠加在全局 prompt 上。
#
#   公式：p_v = global_token + scale * MLP(struct_feat_v)
#
#   这样每个节点得到专属的 structure prompt，让 GNN 在预训练时
#   能区分不同结构角色的节点，提升下游 few-shot 分类的 prototype 质量。
#
# 新增：Domain-Aware Archetypal Analysis (DA-AA)
#   以即插即用方式替换/增强结构提示，通过共享+域特化原型实现：
#   - 多域凸包顶点约束（防坍缩）
#   - 结构-语义隐式解耦
#   - 域公平性（小域表达保证）
#
#   启用方式：--use_daaa 1
# ------------------------------------------------------------


class PrePrompt(nn.Module):
    """预训练 Prompt 模型（含 SCP 支持）。

    Args:
        n_in  : 输入特征维度（PCA 后统一维度）
        n_h   : GNN 隐层维度
        activation : 非线性激活名称
        num_pretrain_dataset_num : 预训练数据集数量
        num_layers_num : GNN 层数
        dropout : GNN dropout
        type_  : prompt 组合类型（'add'/'mul'）
        backbone : 'gcn' 或 'gat'
        alpha  : fea/str 融合系数
        ablation : 消融选项
        use_structure_conditioned_prompt : 是否启用 SCP
        struct_feat_dim : 结构特征维度（1=degree, 3=deg+clust+pagerank）
        struct_prompt_residual : True=残差叠加, False=纯 MLP 输出
        struct_prompt_scale : MLP 输出的缩放系数（防止初期扰动过大）
        random_struct : True=随机打乱结构特征（消融用）
    """

    def __init__(
        self,
        n_in,
        n_h,
        activation,
        num_pretrain_dataset_num,
        num_layers_num,
        dropout,
        type_,
        backbone='gcn',
        alpha=1.0,
        ablation='all',
        # SCP 参数
        use_structure_conditioned_prompt=False,
        struct_feat_dim=1,
        struct_prompt_residual=True,
        struct_prompt_scale=0.05,
        random_struct=False,
        # DA-AA 参数
        use_daaa=False,
        daaa_n_shared=8,
        daaa_n_domain=4,
        daaa_lambda_vert=0.05,
        daaa_lambda_res=0.01,
        daaa_lambda_hsic=0.0,
        daaa_lambda_gc=0.1,
        daaa_max_k_domain=12,
        daaa_use_sparse_coeff=False,
    ):
        super(PrePrompt, self).__init__()

        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_

        # ---- 原始 prompt 参数 ----
        self.feature_prompt_layers = nn.ModuleList(
            [textprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)]
        )
        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)
        ])

        # ---- SCP 参数 ----
        self.use_structure_conditioned_prompt = use_structure_conditioned_prompt
        self.struct_feat_dim = struct_feat_dim
        self.struct_prompt_residual = struct_prompt_residual
        self.struct_prompt_scale = struct_prompt_scale
        self.random_struct = random_struct

        if self.use_structure_conditioned_prompt:
            # 每个数据集 × 每层 一个 MLP：struct_feat -> n_h
            self.structure_prompt_generators = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.struct_feat_dim, n_h),
                        nn.ReLU(),
                        nn.Linear(n_h, n_h),
                    )
                    for _ in range(num_layers_num)
                ])
                for _ in range(num_pretrain_dataset_num)
            ])
            print(
                f"[SCP] enabled | struct_feat_dim={struct_feat_dim} "
                f"residual={struct_prompt_residual} scale={struct_prompt_scale} "
                f"random={random_struct}"
            )
        else:
            self.structure_prompt_generators = None

        # ---- 主干 GNN ----
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if backbone == 'gat':
            self.gcn = GatLayers(n_in, n_h, num_layers_num, dropout)
            str_prompt = [textprompt(n_h * self.gcn.heads, type_) for _ in range(num_layers_num)]
            self.structure_prompt_layers = nn.ModuleList([
                nn.ModuleList(copy.deepcopy(str_prompt))
                for _ in range(num_pretrain_dataset_num)
            ])

        self.combine = alpha
        self.loss = nn.BCEWithLogitsLoss()
        self.ablation_choice = ablation

        # 结构特征缓存（避免每 epoch 重复计算）
        self._feat_cache = {}

        # ---- DA-AA 参数 ----
        self.use_daaa = use_daaa
        if use_daaa:
            self.daaa = DomainAwareAA(
                n_domains=num_pretrain_dataset_num,
                n_shared=daaa_n_shared,
                n_domain=daaa_n_domain,
                dim=n_h,  # 作用在 GNN 输出的节点嵌入上
                lambda_vert=daaa_lambda_vert,
                lambda_res=daaa_lambda_res,
                lambda_hsic=daaa_lambda_hsic,
                lambda_gc=daaa_lambda_gc,
                max_k_domain=daaa_max_k_domain,
                use_sparse_coeff=daaa_use_sparse_coeff,
            )
            print(
                "[DA-AA] enabled | K_s={}, K_d={}, "
                "lv={}, lr={}, lh={}, lg={}".format(
                    daaa_n_shared, daaa_n_domain,
                    daaa_lambda_vert, daaa_lambda_res,
                    daaa_lambda_hsic, daaa_lambda_gc,
                )
            )
        else:
            self.daaa = None

        # DA-AA 辅助损失累积（每 epoch 清零）
        self._daaa_aux_loss = torch.tensor(0.0)
        self._daaa_loss_dict: dict = {}

    # ------------------------------------------------------------------
    # 结构特征计算
    # ------------------------------------------------------------------

    def _degree_structure_feature(self, adj):
        """1-dim 结构特征：log-normalized degree。"""
        if torch.is_tensor(adj) and adj.dim() == 3:
            adj = adj[0]
        if getattr(adj, 'is_sparse', False):
            deg = torch.sparse.sum(adj, dim=1).to_dense()
        else:
            deg = adj.sum(dim=1)
        deg = torch.log1p(deg.float())
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        S = deg.view(-1, 1)
        if self.random_struct:
            idx = torch.randperm(S.shape[0], device=S.device)
            S = S[idx]
        return S

    def _3feat_structure_feature(self, adj):
        """3-dim 结构特征：degree + clustering coefficient + PageRank。"""
        if torch.is_tensor(adj) and adj.dim() == 3:
            adj = adj[0]
        device = adj.device if torch.is_tensor(adj) else torch.device('cpu')

        if adj.is_sparse:
            deg = torch.sparse.sum(adj, dim=1).to_dense()
            adj_dense = adj.to_dense()
        else:
            deg = adj.sum(dim=1)
            adj_dense = adj

        deg = torch.log1p(deg.float())

        # clustering coefficient：diag(A^3) / (d*(d-1))
        tri = torch.diag(torch.matrix_power(adj_dense.float(), 3)) / 2.0
        deg_clamp = deg.float()
        clustering = 2.0 * tri / (deg_clamp * (deg_clamp - 1) + 1e-8)
        clustering[deg_clamp <= 1] = 0.0

        # PageRank（3 次迭代近似）
        adj_cpu = adj_dense.cpu().float()
        N = adj_cpu.shape[0]
        deg_sum_cpu = adj_cpu.sum(dim=1).clamp(min=1e-6)
        adj_norm_cpu = adj_cpu / deg_sum_cpu.unsqueeze(1)
        pr_cpu = torch.ones(N) / N
        for _ in range(3):
            pr_cpu = 0.85 * torch.mv(adj_norm_cpu.T, pr_cpu) + 0.15 / N
        pr = torch.log1p(pr_cpu * N).to(device)

        def norm(x):
            return (x - x.mean()) / (x.std() + 1e-6)

        feat = torch.stack([norm(deg), norm(clustering), norm(pr)], dim=1)
        if self.random_struct:
            idx = torch.randperm(feat.shape[0], device=device)
            feat = feat[idx]
        return feat

    # ------------------------------------------------------------------
    # SCP：构造节点级 structure prompt
    # ------------------------------------------------------------------

    def _build_scp_prompts(self, domain_idx, str_layers, adj):
        """为第 domain_idx 个数据集构造节点级 structure prompt 列表。

        每层的 prompt 形状：[N, n_h]（节点级，不再是全局 [1, n_h]）。

        Args:
            domain_idx : 数据集索引
            str_layers : 该数据集的全局 structure prompt 列表（每层一个 textprompt）
            adj        : 归一化邻接矩阵（用于计算结构特征）

        Returns:
            list of Tensor [N, n_h]，长度 = num_layers_num
        """
        # 缓存结构特征（同一 epoch 内同一数据集只算一次）
        cache_key = (domain_idx, self.struct_feat_dim, self.random_struct)
        if cache_key not in self._feat_cache:
            if self.struct_feat_dim == 1:
                S = self._degree_structure_feature(adj)
            elif self.struct_feat_dim == 3:
                S = self._3feat_structure_feature(adj)
            else:
                raise ValueError(f"struct_feat_dim must be 1 or 3, got {self.struct_feat_dim}")
            self._feat_cache[cache_key] = S.detach()
        else:
            S = self._feat_cache[cache_key]
            if torch.is_tensor(adj):
                S = S.to(device=adj.device)

        conditioned_layers = []
        for layer_idx, global_prompt_layer in enumerate(str_layers):
            # MLP 生成节点级偏移：[N, n_h]
            node_delta = self.structure_prompt_generators[domain_idx][layer_idx](S)
            # 全局 token：[n_h]
            global_token = global_prompt_layer.weight.squeeze(0)

            if self.struct_prompt_residual:
                # 残差叠加：p_v = global_token + scale * delta_v
                node_prompt = global_token.unsqueeze(0) + self.struct_prompt_scale * node_delta
            else:
                # 纯 MLP 输出
                node_prompt = self.struct_prompt_scale * node_delta

            conditioned_layers.append(node_prompt)  # [N, n_h]

        return conditioned_layers

    # ------------------------------------------------------------------
    # 原始 SAMGPT 方法
    # ------------------------------------------------------------------

    def ablation(self, fea_prelogits, str_prelogits):
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits

    def embed(self, seq, adj, sparse, msk, LP, use_domain_prototypes=True):
        h_1 = self.gcn(seq, adj, sparse, LP)
        if self.use_daaa and self.daaa is not None:
            h_1_2d = h_1.squeeze(0) if h_1.dim() == 3 else h_1
            with torch.no_grad():
                if use_domain_prototypes:
                    # 原逻辑（不变）
                    z_s = self.daaa.z_shared.to(h_1_2d.device)
                    a_s = self.daaa._compute_coefficients(h_1_2d, z_s)
                    x_hat = a_s @ z_s
                    h_out = h_1_2d + 0.3 * (x_hat - h_1_2d)
                else:
                    # shared_only：完全不用原型
                    h_out = h_1_2d
            h_1 = h_out.unsqueeze(0) if h_1.dim() == 3 else h_out
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()

    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def get_structure_generators(self):
        """导出 SCP 生成器参数供下游使用（如果启用了 SCP）。"""
        if self.use_structure_conditioned_prompt and self.structure_prompt_generators is not None:
            return [list(gen_list) for gen_list in self.structure_prompt_generators]
        return None

    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers,
                              seq_list, adj_list, sparse=False):
        for fea_pretext, str_layers, seq, adj in zip(
            feature_prompt_layers, structure_prompt_layers, seq_list, adj_list
        ):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)

    def compute_prelogits_GRAPHCL(
        self, feature_prompt_layers, structure_prompt_layers,
        seq_list, adj_list, sparse=False, msk=None,
        samp_bias1=None, samp_bias2=None,
    ):
        for domain_idx, (fea_pretext, str_layers, seq, adj) in enumerate(
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list)
        ):
            if self.ablation_choice == 'None':
                logit = self.graphcledge(
                    self.gcn, seq[0], seq[1], seq[2], seq[3],
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge',
                )
                yield logit
            else:
                # fea 分支：feature prompt
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))]
                fea_prelogits = self.graphcledge(
                    self.gcn,
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3],
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge',
                )

                # str 分支：SCP 或原始全局 prompt
                if self.use_structure_conditioned_prompt:
                    # 节点级 structure prompt（[N, n_h] per layer）
                    str_layers_use = self._build_scp_prompts(domain_idx, str_layers, adj[0])
                else:
                    str_layers_use = str_layers

                str_prelogits = self.graphcledge(
                    self.gcn,
                    seq[0], seq[1], seq[2], seq[3],
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge', str_layers_use,
                )

                # DA-AA 增强：对 GNN 编码后的节点嵌入做原型分析
                if self.use_daaa and self.daaa is not None:
                    h_raw = self.gcn(seq[0], adj[0], sparse, None)
                    h_raw_2d = h_raw.squeeze(0) if h_raw.dim() == 3 else h_raw
                    h_out, aa_losses = self.daaa(h_raw_2d, domain_id=domain_idx)
                    # 梯度隔离：各域损失除以域数量，避免大图梯度主导共享原型
                    n_domains = len(list(self.feature_prompt_layers))
                    self._daaa_aux_loss = self._daaa_aux_loss + aa_losses['total'] / n_domains
                    for k, v in aa_losses.items():
                        key = 'd{}_{}'.format(domain_idx, k)
                        self._daaa_loss_dict[key] = v.item() if torch.is_tensor(v) else v

                yield self.ablation(fea_prelogits, str_prelogits)

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2,
                lbl, samples=None):
        """预训练 forward。每个 epoch 开始前清空结构特征缓存。"""
        # 清空缓存，保证每 epoch 重新计算结构特征
        self._feat_cache.clear()

        # 清空 DA-AA 辅助损失累积
        device = seq_list[0].device
        self._daaa_aux_loss = torch.tensor(0.0, device=device)
        self._daaa_loss_dict = {}

        total_loss = torch.tensor(0.0, dtype=torch.float32).to(device)

        if samples is None:
            logits = list(self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers, self.structure_prompt_layers,
                seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2,
            ))
            for i in range(len(logits)):
                total_loss += self.loss(logits[i], lbl[i])
        else:
            logits = list(self.compute_prelogits_LP(
                self.feature_prompt_layers, self.structure_prompt_layers,
                seq_list, adj_list, sparse,
            ))
            if type(samples) == list:
                samples = [
                    torch.tensor(s, dtype=torch.int64).to(device)
                    for s in samples
                ]
                for i in range(len(logits)):
                    total_loss += compareloss(logits[i], samples[i], temperature=1)
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)

        # 加入 DA-AA 辅助损失
        if self.use_daaa and self._daaa_aux_loss.requires_grad:
            total_loss = total_loss + self._daaa_aux_loss

        return total_loss


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq


def svd_compression(seq, k):
    U, Sigma, VT = np.linalg.svd(seq)
    return U[:, :k].dot(np.diag(Sigma[:k]))


def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten().reshape(-1, 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples)).to(feature.device).reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:].permute(1, 0).sum(dim=1, keepdim=True)
    return (-1 * torch.log(numerator / denominator)).mean()


def prompt_pretrain_sample(adj, n):
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i]:indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1:1 + n] = zero_index_i_row[0:n]
    return torch.tensor(res.astype(int))
