import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, GcnLayers, MLP, GatLayers
from layers import AvgReadout 
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from layers.prompt import *
import copy
import logging
from utils.joint_bmm import compute_fn_posterior_from_pairs


logger = logging.getLogger(__name__)
_GRAPHCL_JOINT_LOG_COUNT = 0

# ------------------------------------------------------------
# preprompt.py：预训练阶段的核心模型 PrePrompt。
#
# 该模块的要点：
# 1) 预训练任务：GRAPHCL，需要两份增强视图 + DGI/GraphCL 类对比损失（BCEWithLogitsLoss）
#
# 2) prompt 分为两种：
#    - feature_prompt_layers：对输入特征 seq 做 add/mul 调制
#    - structure_prompt_layers：作为“层 prompt”传给 GNN 的每一层（num_layers_num 层）
#
# 3) 多数据集预训练：对每个 pretrain dataset 都有一套 prompt 参数（ModuleList 按数据集索引）。
#
# 4) ablation：控制仅使用 fea prompting / 仅使用 str prompting / 两者融合等。
# ------------------------------------------------------------


class PrePrompt(nn.Module):
    """预训练 Prompt 模型。

    Args:
        n_in: 输入特征维度（PCA 后/统一维度 unify_dim）
        n_h: 隐层维度（GNN 输出维度）
        activation: 非线性（GraphCL/DGI 内部可能用到）
        num_pretrain_dataset_num: 预训练数据集数量（或参与预训练的图数量）
        num_layers_num: GNN 层数
        dropout: GNN dropout
        type_: prompt 的组合类型（'add'/'mul'），传给 textprompt
        backbone: 'gcn' 或 'gat'
        alpha: fea/str 融合系数（combine）
        ablation: 消融选项（'all','ft','st','None',...）
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
        joint_bmm_topk=0,
        ablation='all',
    ):
        super(PrePrompt, self).__init__()

        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)

        # 图级 readout（平均池化）
        self.read = AvgReadout()

        self.prompttype = type_

        # ------------------- prompt 参数（按数据集分组）-------------------
        # 每个数据集一套 feature prompt（形状 [1, n_in]）
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)])

        # 每个数据集一套 structure prompt：每层一个 textprompt（通常为 [1, n_h]）
        self.structure_prompt_layers = nn.ModuleList(
            [nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)]) for _ in range(num_pretrain_dataset_num)]
        )

        # ------------------- 主干 GNN -------------------
        # 默认使用 GCN；可选 GAT
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if backbone == 'gat':
            self.gcn = GatLayers(n_in, n_h, num_layers_num, dropout)

            # GAT 多头时，每层输出维度通常变成 n_h * heads（concat=True 的情形）
            str_prompt = [textprompt(n_h * self.gcn.heads, type_) for _ in range(num_layers_num)]
            # str_prompt.append(textprompt(n_h, type_))

            # 重新构造 structure_prompt_layers：每个数据集拷贝一份同构 prompt 列表
            self.structure_prompt_layers = nn.ModuleList([nn.ModuleList(copy.deepcopy(str_prompt)) for _ in range(num_pretrain_dataset_num)])

        # combine: 融合 fea/str 的系数（在 ablation('all') 下生效）
        self.combine = alpha
        self.joint_bmm_topk = joint_bmm_topk

        # GRAPHCL/DGI 这类任务常用 BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

    def ablation(self, fea_prelogits, str_prelogits):
        """根据 ablation_choice 组合 fea/str 两类 logits."""
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits

    def compute_prelogits_GRAPHCL(
        self,
        feature_prompt_layers,
        structure_prompt_layers,
        seq_list,
        adj_list,
        sparse=False,
        msk=None,
        samp_bias1=None,
        samp_bias2=None,
        return_intermediate=False,
    ):
        """计算 GRAPHCL 预训练 logits（逐数据集 yield）。

        seq_list 中的每个 seq 预期是一个 list/stack：
            [feature, shuf_fts, feature.detach(), feature.detach()]
        adj_list 中每个 adj 预期是 stack：
            [adj, aug_adj1, aug_adj2]
        这些格式来自 utils/aug.py 的 build_aug()。

        当 ablation_choice != 'None' 时，会分别计算：
        - fea_prelogits：对 seq 的每个视图做 feature prompt 后喂入 GraphCL
        - str_prelogits：将 structure prompt 传入 gcn，计算结构提示下的 GraphCL
        """
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.graphcledge(
                    self.gcn,
                    seq[0],
                    seq[1],
                    seq[2],
                    seq[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    'edge',
                    return_intermediate=return_intermediate,
                )
            else:
                # 对每个视图都应用 feature prompt
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))]
                fea_prelogits = self.graphcledge(
                    self.gcn,
                    preseq_list[0],
                    preseq_list[1],
                    preseq_list[2],
                    preseq_list[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    aug_type='edge',
                    return_intermediate=return_intermediate,
                )

                # 结构 prompt：不改输入 seq，但把 prompt 列表传给 gcn
                str_prelogits = self.graphcledge(
                    self.gcn,
                    seq[0],
                    seq[1],
                    seq[2],
                    seq[3],
                    adj[0],
                    adj[1],
                    adj[2],
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                    'edge',
                    str_layers,
                    return_intermediate=return_intermediate,
                )
                if return_intermediate:
                    fea_ret, h_pos, h_neg = fea_prelogits
                    str_ret = str_prelogits[0] if isinstance(str_prelogits, tuple) else str_prelogits
                    yield self.ablation(fea_ret, str_ret), h_pos, h_neg
                else:
                    yield self.ablation(fea_prelogits, str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        """得到节点 embedding h_1 及图级向量 c（readout）。

        返回 detach 后的张量，通常用于下游评测（不反传梯度）。
        """
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        """导出 prompt 参数（detach）供下游 downprompt 使用."""
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [[layer.weight.detach() for layer in structure_prompt_layer] for structure_prompt_layer in self.structure_prompt_layers]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl):
        return self.forward_with_joint_bmm(seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl, joint_bmm_enabled=True)

    def forward_with_joint_bmm(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl, joint_bmm_enabled=True):
        """预训练 forward：仅 GraphCL 路径。

        Args:
            seq_list/adj_list: 预训练数据列表（多数据集）
            lbl: GRAPHCL 的对比标签（list，与 seq_list 对齐）
            joint_bmm_enabled: 是否启用 joint_bmm 负项加权
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        if not joint_bmm_enabled:
            logits = list(
                self.compute_prelogits_GRAPHCL(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    sparse,
                    msk,
                    samp_bias1,
                    samp_bias2,
                )
            )
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i])
                total_loss += loss
            return total_loss

        logits = list(
            self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers,
                self.structure_prompt_layers,
                seq_list,
                adj_list,
                sparse,
                msk,
                samp_bias1,
                samp_bias2,
                return_intermediate=True,
            )
        )
        global _GRAPHCL_JOINT_LOG_COUNT
        for i in range(len(logits)):
            ret, h_pos, h_neg = logits[i]
            neg_weight, p_fn = self._compute_graphcl_joint_weights(h_pos, h_neg, adj_list[i])

            lbl_i = lbl[i]
            if lbl_i.dim() == 2 and lbl_i.shape[0] == 1:
                lbl_i = lbl_i.squeeze(0)
            loss_vec = F.binary_cross_entropy_with_logits(ret.squeeze(0), lbl_i, reduction='none')
            weight = torch.ones_like(loss_vec)
            n_nodes = neg_weight.shape[0]
            weight[n_nodes:] = neg_weight
            loss = (loss_vec * weight).mean()
            total_loss += loss

            if _GRAPHCL_JOINT_LOG_COUNT < 3:
                logger.info(
                    '[joint_bmm][GRAPHCL] dataset_idx=%d, topk=%s, ret_shape=%s, neg_weight[min=%.4f, mean=%.4f, max=%.4f], p_fn[min=%.4f, mean=%.4f, max=%.4f]',
                    i,
                    'all' if self.joint_bmm_topk <= 0 else self.joint_bmm_topk,
                    tuple(ret.shape),
                    float(neg_weight.min().item()),
                    float(neg_weight.mean().item()),
                    float(neg_weight.max().item()),
                    float(p_fn.min().item()),
                    float(p_fn.mean().item()),
                    float(p_fn.max().item()),
                )
        _GRAPHCL_JOINT_LOG_COUNT += 1

        return total_loss

    def _compute_graphcl_joint_weights(self, h_pos, h_neg, adj):
        """基于节点候选对的后验概率，为 GraphCL 的负项生成权重。"""
        node_num = h_pos.shape[1]
        candidate_k = node_num - 1 if self.joint_bmm_topk is None or self.joint_bmm_topk <= 0 else min(self.joint_bmm_topk, node_num - 1)

        z = F.normalize(h_pos.squeeze(0), p=2, dim=1)
        sim_mat = torch.matmul(z, z.t())
        sim_mat.fill_diagonal_(-1.0)
        hard_idx = torch.topk(sim_mat, k=candidate_k, dim=1).indices
        hard_sim = torch.gather(sim_mat, 1, hard_idx)

        if isinstance(adj, (list, tuple)):
            adj = adj[0]
        if isinstance(adj, torch.Tensor) and adj.dim() > 2:
            adj = adj[0]

        p_fn = compute_fn_posterior_from_pairs(hard_sim, hard_idx, adj=adj)
        p_fn = torch.tensor(p_fn, dtype=torch.float32, device=h_pos.device)
        neg_weight = 1.0 - p_fn.mean(dim=1)
        return neg_weight, p_fn



def pca_compression(seq, k):
    """PCA 降维到 k 维，并打印累计解释方差比例."""
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)

    print(pca.explained_variance_ratio_.sum())
    return seq


def svd_compression(seq, k):
    """SVD 压缩（保留前 k 个奇异值）。"""
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:, :k].shape)
    print(VT[:k, :].shape)
    res = U[:, :k].dot(np.diag(Sigma[:k]))

    return res


