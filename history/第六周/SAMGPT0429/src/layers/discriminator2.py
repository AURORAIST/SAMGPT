import torch
import torch.nn as nn


class Discriminator2(nn.Module):
    """一个更“精简版”的判别器（对比 `discriminator.py` 的更健壮实现）。

    核心仍然是使用 `nn.Bilinear` 计算正/负样本得分：
      score = h^T W c + b

    与常见 DGI 判别器的差异点：
    - 这里在 forward 中直接使用 `c_x = c`，而不是把 `c` broadcast/expand 成与 `h_pl` 同形状。
      因此它隐含要求：传入的 `c` 的形状/维度应当能与 `h_pl` 在 `nn.Bilinear` 的广播规则下匹配。
    """

    def __init__(self, n_h):
        """初始化。

        Args:
            n_h: 隐向量维度（节点 embedding 与上下文向量的特征维）。
        """
        super(Discriminator2, self).__init__()
        # 双线性打分层：输入 (n_h, n_h) -> 输出 1 维得分
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        # 初始化 Bilinear 权重
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """仅对 Bilinear 做 Xavier 初始化，bias 置 0。"""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """前向打分。

        常见（但取决于调用方）形状约定：
        - h_pl / h_mi: [B, N, F]
        - c: 可能是 [B, N, F] 或可广播到 [B, N, F] 的形状

        说明：
        - 原作者曾考虑将 `c` 先 `unsqueeze` 后 `expand_as(h_pl)`（见注释掉的两行），
          这对应“图级向量 c: [B,F] 广播到每个节点”的经典写法。
        - 但当前实现使用 `c_x = c`，因此如果 `c` 仍是 [B,F]，则需要依赖 `nn.Bilinear`
          对批维/额外维度的支持或由上游提前把 `c` 处理成与节点表示对齐的形状。

        Args:
            c: 上下文向量或已对齐的上下文张量（需与 `h_pl` 在 bilinear 计算时可匹配）。
            h_pl: 正样本节点表示。
            h_mi: 负样本节点表示。
            s_bias1/s_bias2: 可选 bias，分别加到正/负样本得分上。

        Returns:
            logits: 将正负得分在节点维拼接后的结果，通常形状为 [B, 2N]。
        """
        # 经典写法（被注释掉）：
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        # 当前写法：直接使用 c，要求其形状已与 h_pl/h_mi 对齐或可广播匹配。
        c_x = c

        # f_k 输出形状通常为 [..., 1]，这里 squeeze 掉最后一维得到 [...]
        # 若输入为 [B,N,F]，则 sc_* 通常为 [B,N]
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        # 可选 bias 加成
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # 拼接正/负得分： [B,N] -> [B,2N]
        logits = torch.cat((sc_1, sc_2), 1)

        return logits

