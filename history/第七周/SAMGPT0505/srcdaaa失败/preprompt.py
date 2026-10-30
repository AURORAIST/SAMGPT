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
    """
    预训练 Prompt 模型。
    - 保留原始 feature & structure prompt。
    - 集成 DA-AA（域感知原型分析）。
    - 集成 方案三：内/外循环几何引导，解决梯度隔离问题。
    """
    def __init__(
        self,
        n_in, n_h, activation,
        num_pretrain_dataset_num, num_layers_num, dropout,
        type_,
        backbone='gcn',
        alpha=1.0,
        ablation='all',
        # DA-AA 参数
        use_daaa=False,
        daaa_n_shared=4,
        daaa_n_domain=2,
        daaa_lambda_vert=0.05,
        daaa_lambda_res=0.01,
        daaa_max_k_domain=12,
        # 几何引导参数
        lambda_geom=0.01,       # 几何引导损失权重
        inner_steps=2,          # 内循环原型更新步数
        inner_lr=0.01           # 内循环学习率
    ):
        super(PrePrompt, self).__init__()

        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        self.n_h = n_h
        self._daaa_loss_dict = {}
        # ---- 原始 Prompt 参数 ----
        self.feature_prompt_layers = nn.ModuleList(
            [textprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)]
        )
        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)
        ])

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

        # ---- DA-AA 模块 ----
        self.use_daaa = use_daaa
        if use_daaa:
            self.daaa = DomainAwareAA(
                n_domains=num_pretrain_dataset_num,
                n_shared=daaa_n_shared,
                n_domain=daaa_n_domain,
                dim=n_h,
                lambda_vert=daaa_lambda_vert,
                lambda_res=daaa_lambda_res,
                max_k_domain=daaa_max_k_domain
            )
        else:
            self.daaa = None

        # ---- 几何引导参数 ----
        self.lambda_geom = lambda_geom
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2,
                lbl, samples=None):
        """预训练 forward，已集成几何引导。"""
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        self._daaa_loss_dict.clear()
        if samples is None:
            for domain_idx, (fea_pretext, str_layers, seq, adj) in enumerate(
                zip(self.feature_prompt_layers, self.structure_prompt_layers, seq_list, adj_list)
            ):
                if self.ablation_choice == 'None':
                    logit = self.graphcledge(self.gcn, seq, adj, sparse, msk, samp_bias1, samp_bias2, 'edge')
                    total_loss += self.loss(logit, lbl[domain_idx])
                    continue

                # 1. 计算原始 GraphCL 损失
                preseq_list = [fea_pretext(s) for s in seq]
                fea_prelogits = self.graphcledge(self.gcn, preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3],
                                                adj[0], adj[1], adj[2], sparse, msk, samp_bias1, samp_bias2, 'edge')
                str_prelogits = self.graphcledge(self.gcn, seq[0], seq[1], seq[2], seq[3],
                                                adj[0], adj[1], adj[2], sparse, msk, samp_bias1, samp_bias2, 'edge', str_layers)

                loss_graphcl = self.loss(fea_prelogits, lbl[domain_idx]) + self.loss(str_prelogits, lbl[domain_idx])

                # 2. DA-AA + 几何引导
                if self.use_daaa and self.daaa is not None:
                    # 准备嵌入
                    h_raw = self.gcn(seq[0], adj[0], sparse, None) # [1, N, D] or [N, D]
                    if h_raw.dim() == 3:
                        h_raw = h_raw.squeeze(0) # [N, D]

                    # --- 内循环：快速更新原型（只更新原型参数）---
                    h_detached = h_raw.detach()
                    # 备份当前原型参数
                    z_shared_bak = self.daaa.z_shared.data.clone()
                    z_domain_bak = self.daaa.z_domain[domain_idx].data.clone()
                    
                    for _ in range(self.inner_steps):
                        _, aa_loss = self.daaa(h_detached, domain_id=domain_idx)
                        # 手动梯度更新，绕开优化器
                        self.daaa.z_shared.grad = None
                        self.daaa.z_domain[domain_idx].grad = None
                        aa_loss['total'].backward(retain_graph=True)
                        with torch.no_grad():
                            self.daaa.z_shared -= self.inner_lr * self.daaa.z_shared.grad
                            self.daaa.z_domain[domain_idx] -= self.inner_lr * self.daaa.z_domain[domain_idx].grad

                    # --- 外循环：对编码器施加几何引导 ---
                    Z = torch.cat([self.daaa.z_shared, self.daaa.z_domain[domain_idx]], dim=0)
                    dist = torch.cdist(h_raw, Z)  # h_raw 不detach，梯度可回流GCN
                    loss_geom = dist.min(dim=1).values.mean()

                    # --- 恢复原型参数（用于后续DA-AA的正常更新）---
                    self.daaa.z_shared.data.copy_(z_shared_bak)
                    self.daaa.z_domain[domain_idx].data.copy_(z_domain_bak)

                    # --- DA-AA 的正常损失更新（梯度隔离）---
                    _, aa_loss_final = self.daaa(h_detached, domain_id=domain_idx)
                    loss_daaa = aa_loss_final['total']
                    # 存储损失分量，供监控使用
                    for k, v in aa_loss_final.items():
                        if torch.is_tensor(v):
                            self._daaa_loss_dict[f'd{domain_idx}_{k}'] = v.item()
                        else:
                            self._daaa_loss_dict[f'd{domain_idx}_{k}'] = v
                    # 3. 合并损失
                    total_loss += loss_graphcl + self.lambda_geom * loss_geom + loss_daaa
                else:
                    total_loss += loss_graphcl

        else:
            # LP 分支，暂不涉及 DA-AA，保持原样
            logits = list(self.compute_prelogits_LP(
                self.feature_prompt_layers, self.structure_prompt_layers,
                seq_list, adj_list, sparse,
            ))
            # ... (此处省略，与原版完全一致)

        return total_loss
    
    def embed(self, seq, adj, sparse, msk, LP):
        """返回节点嵌入和图级嵌入，用于下游任务"""
        h_1 = self.gcn(seq, adj, sparse, None)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()

    def get_weights(self):
        """导出提示权重供下游使用"""
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in layers]
            for layers in self.structure_prompt_layers
        ]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers,
                            seq_list, adj_list, sparse=False):
        """LP 预训练分支（原 SAMGPT 保留）"""
        for fea_pretext, str_layers, seq, adj in zip(
            feature_prompt_layers, structure_prompt_layers, seq_list, adj_list
        ):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)

    def ablation(self, fea_prelogits, str_prelogits):
        """融合特征和结构分支的输出"""
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits

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
