import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers
from layers import AvgReadout 
import tqdm
import numpy as np
import logging
import os
from sklearn.decomposition import PCA
from layers.prompt import *
import copy

# ------------------------------------------------------------
# preprompt.py：预训练阶段的核心模型 PrePrompt。
#
# 该模块的要点：
# 1) 预训练任务支持两类：
#    - GRAPHCL：需要两份增强视图 + DGI/GraphCL 类对比损失（BCEWithLogitsLoss）
#    - LP / splitLP：基于负采样 tuples 的对比损失 compareloss（自定义）
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
        ablation='all',
        reliability_loss=True,
        reliability_mode='embedding',
        reliability_visualize=True,
        reliability_visual_dir=None,
        reliability_visual_interval=500,
        reliability_log_interval=1000,
    ):
        super(PrePrompt, self).__init__()

        # 两类预训练目标的封装
        self.lp = Lp(n_in, n_h)
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

        # GRAPHCL/DGI 这类任务常用 BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_none = nn.BCEWithLogitsLoss(reduction='none')
        self.reliability_loss = reliability_loss
        self.reliability_mode = reliability_mode
        self.reliability_log_interval = reliability_log_interval
        self._reliability_forward_count = 0
        self.reliability_visualize = reliability_visualize
        self.reliability_visual_dir = reliability_visual_dir
        self.reliability_visual_interval = reliability_visual_interval

        self.ablation_choice = ablation

    def ablation(self, fea_prelogits, str_prelogits):
        """根据 ablation_choice 组合 fea/str 两类 logits."""
        if isinstance(fea_prelogits, tuple) or isinstance(str_prelogits, tuple):
            if isinstance(fea_prelogits, tuple):
                fea_logits, fea_weights = fea_prelogits
            else:
                fea_logits, fea_weights = fea_prelogits, None

            if isinstance(str_prelogits, tuple):
                str_logits, str_weights = str_prelogits
            else:
                str_logits, str_weights = str_prelogits, None

            if self.ablation_choice == 'st':
                logits = str_logits
                weights = str_weights if str_weights is not None else fea_weights
            elif self.ablation_choice == 'ft':
                logits = fea_logits
                weights = fea_weights if fea_weights is not None else str_weights
            else:
                logits = fea_logits + self.combine * str_logits
                if fea_weights is not None and str_weights is not None:
                    weights = 0.5 * (fea_weights + str_weights)
                elif fea_weights is not None:
                    weights = fea_weights
                else:
                    weights = str_weights
            return logits, weights.detach()

        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits

    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, sparse=False):
        """计算 LP/splitLP 预训练的 logits（逐数据集 yield）。

        对每个预训练数据集：
        - fea_prelogits：先对 seq 做 feature prompt，再做 LP 任务
        - str_prelogits：把 structure prompt 传入 gcn（影响每层），再做 LP 任务
        - 最终通过 ablation() 得到融合 logits
        """
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)

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
                    return_reliability=self.reliability_loss,
                    reliability_mode=self.reliability_mode,
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
                    return_reliability=False,
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
                    return_reliability=self.reliability_loss,
                    reliability_mode=self.reliability_mode,
                )

                yield self.ablation(fea_prelogits, str_prelogits)

    @staticmethod
    def _quantile_text(values):
        values = values.detach().reshape(-1).float()
        qs = torch.quantile(values, values.new_tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
        return '/'.join(f'{v.item():.4f}' for v in qs)

    @staticmethod
    def _hist_overlap(left, right, bins=50, low=0.0, high=1.0):
        left = left.detach().reshape(-1).float()
        right = right.detach().reshape(-1).float()
        if left.numel() == 0 or right.numel() == 0:
            return float('nan')
        left_hist = torch.histc(left.clamp(low, high), bins=bins, min=low, max=high)
        right_hist = torch.histc(right.clamp(low, high), bins=bins, min=low, max=high)
        left_hist = left_hist / left_hist.sum().clamp_min(1e-12)
        right_hist = right_hist / right_hist.sum().clamp_min(1e-12)
        return torch.minimum(left_hist, right_hist).sum().item()

    def _log_reliability_stats(self, reliability_weights, base_losses=None, weighted_losses=None, diagnostics=None):
        if not self.reliability_loss or len(reliability_weights) == 0:
            return

        self._reliability_forward_count += 1
        if self._reliability_forward_count != 1 and self._reliability_forward_count % self.reliability_log_interval != 0:
            return

        pos_parts = []
        neg_parts = []
        for weights in reliability_weights:
            weights = weights.detach()
            half = weights.shape[1] // 2
            pos_parts.append(weights[:, :half].reshape(-1))
            neg_parts.append(weights[:, half:].reshape(-1))

        pos = torch.cat(pos_parts)
        neg = torch.cat(neg_parts)
        msg = (
            f"[ReliabilityLoss] step={self._reliability_forward_count} "
            f"mode={self.reliability_mode} "
            f"pos(mean/min/max)={pos.mean().item():.4f}/{pos.min().item():.4f}/{pos.max().item():.4f} "
            f"neg(mean/min/max)={neg.mean().item():.4f}/{neg.min().item():.4f}/{neg.max().item():.4f}"
        )
        if base_losses is not None and weighted_losses is not None and len(base_losses) > 0:
            base_loss = torch.stack(base_losses).mean()
            weighted_loss = torch.stack(weighted_losses).mean()
            ratio = weighted_loss / (base_loss + 1e-12)
            msg += (
                f" base_bce={base_loss.item():.6f} "
                f"weighted_bce={weighted_loss.item():.6f} "
                f"weighted/base={ratio.item():.4f}"
            )
        if diagnostics is not None and len(diagnostics) > 0:
            diag = self._merge_reliability_diagnostics(diagnostics)
            logit_overlap = self._hist_overlap(diag['pos_logits'], diag['neg_logits'], bins=80, low=-20.0, high=20.0)
            loss_overlap = self._hist_overlap(diag['pos_weighted_loss'], diag['neg_weighted_loss'], bins=80, low=0.0, high=1.0)
            msg += (
                f" rho_q05/25/50/75/95_pos={self._quantile_text(pos)}"
                f" rho_q05/25/50/75/95_neg={self._quantile_text(neg)}"
                f" logit_mean_pos/neg={diag['pos_logits'].mean().item():.4f}/{diag['neg_logits'].mean().item():.4f}"
                f" prob_mean_pos/neg={diag['pos_probs'].mean().item():.4f}/{diag['neg_probs'].mean().item():.4f}"
                f" weighted_loss_mean_pos/neg={diag['pos_weighted_loss'].mean().item():.6f}/{diag['neg_weighted_loss'].mean().item():.6f}"
                f" overlap_logit/loss={logit_overlap:.4f}/{loss_overlap:.4f}"
            )
        print(msg)
        logging.info(msg)
        self._save_reliability_visual(pos, neg, diagnostics)

    @staticmethod
    def _merge_reliability_diagnostics(diagnostics):
        keys = diagnostics[0].keys()
        return {key: torch.cat([item[key].detach().reshape(-1) for item in diagnostics]) for key in keys}

    @staticmethod
    def _build_reliability_diagnostics(logit, label, weights, weighted_loss_per_entry):
        with torch.no_grad():
            logit = logit.detach()
            label = label.detach().to(logit.device)
            weights = weights.detach().to(logit.device)
            weighted_loss_per_entry = weighted_loss_per_entry.detach()

            pos_mask = label > 0.5
            neg_mask = ~pos_mask
            probs = torch.sigmoid(logit)

            return {
                'pos_logits': logit[pos_mask],
                'neg_logits': logit[neg_mask],
                'pos_probs': probs[pos_mask],
                'neg_probs': probs[neg_mask],
                'pos_weights': weights[pos_mask],
                'neg_weights': weights[neg_mask],
                'pos_weighted_loss': weighted_loss_per_entry[pos_mask],
                'neg_weighted_loss': weighted_loss_per_entry[neg_mask],
            }

    def _save_reliability_visual(self, pos, neg, diagnostics=None):
        if not self.reliability_visualize or self.reliability_visual_dir is None:
            return
        if self._reliability_forward_count != 1 and self._reliability_forward_count % self.reliability_visual_interval != 0:
            return
        try:
            os.makedirs(self.reliability_visual_dir, exist_ok=True)
            os.environ.setdefault('MPLCONFIGDIR', os.path.join(self.reliability_visual_dir, '.matplotlib'))
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            logging.info(f"[ReliabilityLoss] skip visualization: {exc}")
            return

        pos_np = pos.detach().float().cpu().numpy()
        neg_np = neg.detach().float().cpu().numpy()

        diag = self._merge_reliability_diagnostics(diagnostics) if diagnostics is not None and len(diagnostics) > 0 else None
        if diag is None:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
            axes = [ax]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
            axes = axes.reshape(-1)
            ax = axes[0]
        ax.hist(pos_np, bins=30, range=(0.0, 1.0), density=True, alpha=0.65, label='positive rho', color='#2f80ed')
        ax.hist(neg_np, bins=30, range=(0.0, 1.0), density=True, alpha=0.65, label='negative true prob', color='#eb5757')
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel('rho')
        ax.set_ylabel('density')
        ax.set_title(f'Reliability rho ({self.reliability_mode}) step {self._reliability_forward_count}')
        ax.legend(frameon=False)

        if diag is not None:
            pos_logits = diag['pos_logits'].float().cpu().numpy()
            neg_logits = diag['neg_logits'].float().cpu().numpy()
            pos_probs = diag['pos_probs'].float().cpu().numpy()
            neg_probs = diag['neg_probs'].float().cpu().numpy()
            pos_loss = diag['pos_weighted_loss'].float().cpu().numpy()
            neg_loss = diag['neg_weighted_loss'].float().cpu().numpy()

            ax = axes[1]
            ax.hist(pos_logits, bins=60, density=True, alpha=0.65, label='positive logits', color='#2f80ed')
            ax.hist(neg_logits, bins=60, density=True, alpha=0.65, label='negative logits', color='#eb5757')
            ax.set_xlabel('logit')
            ax.set_ylabel('density')
            ax.set_title('Discriminator logits')
            ax.legend(frameon=False)

            ax = axes[2]
            ax.hist(pos_probs, bins=40, range=(0.0, 1.0), density=True, alpha=0.65, label='positive sigmoid', color='#2f80ed')
            ax.hist(neg_probs, bins=40, range=(0.0, 1.0), density=True, alpha=0.65, label='negative sigmoid', color='#eb5757')
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel('sigmoid(logit)')
            ax.set_ylabel('density')
            ax.set_title('Prediction probability')
            ax.legend(frameon=False)

            ax = axes[3]
            ax.hist(pos_loss, bins=50, density=True, alpha=0.65, label='positive weighted BCE', color='#2f80ed')
            ax.hist(neg_loss, bins=50, density=True, alpha=0.65, label='negative weighted BCE', color='#eb5757')
            ax.set_xlabel('weighted BCE contribution')
            ax.set_ylabel('density')
            ax.set_title('Loss contribution')
            ax.legend(frameon=False)
        fig.tight_layout()

        out_path = os.path.join(
            self.reliability_visual_dir,
            f'reliability_{self.reliability_mode}_step_{self._reliability_forward_count:06d}.png',
        )
        fig.savefig(out_path)
        plt.close(fig)
        logging.info(f"[ReliabilityLoss] saved visualization: {out_path}")

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

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl, samples=None):
        """预训练 forward：根据是否提供 samples 来区分 GRAPHCL 与 LP。

        Args:
            seq_list/adj_list: 预训练数据列表（多数据集）
            lbl: GRAPHCL 的对比标签（list，与 seq_list 对齐）
            samples: LP/splitLP 的负采样 tuples
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        # GRAPHCL 路径：samples==None
        if samples == None:
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
            reliability_weights = []
            base_losses = []
            weighted_losses = []
            diagnostics = []
            for i in range(len(logits)):
                if isinstance(logits[i], tuple):
                    logit, weights = logits[i]
                    reliability_weights.append(weights)
                    loss_per_entry = self.loss_none(logit, lbl[i])
                    base_loss = loss_per_entry.mean()
                    weights = weights.to(logit.device)
                    weighted_loss_per_entry = loss_per_entry * weights
                    loss = weighted_loss_per_entry.mean()
                    base_losses.append(base_loss.detach())
                    weighted_losses.append(loss.detach())
                    diagnostics.append(self._build_reliability_diagnostics(logit, lbl[i], weights, weighted_loss_per_entry))
                else:
                    loss = self.loss(logits[i], lbl[i])
                total_loss += loss
            self._log_reliability_stats(reliability_weights, base_losses, weighted_losses, diagnostics)

        # LP / splitLP 路径
        else:
            logits = list(
                self.compute_prelogits_LP(
                    self.feature_prompt_layers,
                    self.structure_prompt_layers,
                    seq_list,
                    adj_list,
                    sparse,
                )
            )

            # splitLP：samples 是 list（每个数据集各自一份 tuples）
            if type(samples) == list:
                samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device) for sample in samples]
                for i in range(len(logits)):
                    loss = compareloss(logits[i], samples[i], temperature=1)
                    total_loss += loss

            # LP：samples 是一个整体 tuples（来自合并图）
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)

        return total_loss



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


def mygather(feature, index):
    """辅助函数：按 index 从 feature 中 gather，并保持 batch 结构。

    Args:
        feature: Tensor [N, F]
        index: Tensor [B, K]（或可 reshape 成类似结构）

    Returns:
        Tensor [B, K, F]
    """
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    """一个基于 cosine similarity 的对比损失（与 prompt_pretrain_sample 生成的 tuples 配套）。

    tuples 的每行：
        [pos, neg1, neg2, ...]
    其中：
        - h_i: anchor（按行号 i 取 feature[i]）
        - h_tuples: 从 tuples 中 gather 的 pos/neg 表示
    损失形式接近 InfoNCE：
        -log( exp(sim(anchor,pos))/sum exp(sim(anchor,neg_j)) )

    注意：实现里 exp(sim)/temperature 的写法与常见 exp(sim/temperature) 不同，保持原样不改。
    """
    h_tuples = mygather(feature, tuples)

    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)

    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj, n):
    """为 LP/splitLP 生成负采样 tuples。

    对每个节点 i：
    - 从其非零邻居中随机取 1 个作为“正样本”(pos)
      若无邻居则 pos=i
    - 从其零邻居（非相连节点）中随机取 n 个作为负样本

    Returns:
        Tensor [N, 1+n]，每行 [pos, neg1..negn]

    要求：adj 为 scipy.sparse.csr_matrix（使用 indices/indptr）。
    """
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))

    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i] : indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1 : 1 + n] = zero_index_i_row[0:n]

    return torch.tensor(res.astype(int))
