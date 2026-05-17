import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """DGI / 对比学习判别器。

    该模块用一个双线性打分函数 `nn.Bilinear` 来计算：
    - 正样本 (h_pl, c) 的相关性得分
    - 负样本 (h_mi, c) 的相关性得分

    其中：
    - `h_pl`/`h_mi` 通常是节点表示 (positive/negative) ：形状 [B, N, F]
    - `c` 通常是图级/上下文向量：形状 [B, F]

    最终输出把正负得分在节点维度上拼接成 logits：形状 [B, 2N]
    （实现中如果 B=1 也会得到 [1, 2N]）。
    """

    def __init__(self, n_h):
        """初始化判别器。

        Args:
            n_h: 隐向量维度 F（节点表示和上下文向量的特征维）。
        """
        super(Discriminator, self).__init__()
        self.n_h = n_h
        # 双线性层：对 (x, y) 做 x^T W y + b，输出标量（1 维）。
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        # 遍历所有子模块并做参数初始化。
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """对 Bilinear 权重做 Xavier 初始化，bias 置 0。"""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """前向计算。

        Args:
            c: 上下文/图级向量，期望 [B, F]；若传入 [F] 会自动升维到 [1, F]。
            h_pl: 正样本节点表示，期望 [B, N, F]；若传入 [N, F] 会自动升维到 [1, N, F]。
            h_mi: 负样本节点表示，期望 [B, N, F]；若传入 [N, F] 会自动升维到 [1, N, F]。
            s_bias1/s_bias2: 可选的结构 bias（如邻接相关的偏置项），会分别加到正/负得分上。

        Returns:
            logits: 将正负得分拼接后的张量，形状 [B, 2N]。
        """
        # 兼容用户传入不带 batch 维的情况：
        # - h_pl/h_mi: [N,F] -> [1,N,F]
        # - c: [F] -> [1,F]
        # Expected shapes (typical): h_pl/h_mi [B,N,n_h], c [B,n_h]
        if h_pl.dim() == 2:
            h_pl = h_pl.unsqueeze(0)
        if h_mi.dim() == 2:
            h_mi = h_mi.unsqueeze(0)
        if c.dim() == 1:
            c = c.unsqueeze(0)

        # 形状检查：确保后续 view/expand 与 bilinear 计算安全。
        if h_pl.dim() != 3 or h_mi.dim() != 3:
            raise ValueError(f"h_pl/h_mi must be 3D [B,N,F], got h_pl={tuple(h_pl.shape)}, h_mi={tuple(h_mi.shape)}")
        if c.dim() != 2:
            raise ValueError(f"c must be 2D [B,F], got c={tuple(c.shape)}")
        if h_pl.shape != h_mi.shape:
            raise ValueError(f"h_pl and h_mi shapes must match, got {tuple(h_pl.shape)} vs {tuple(h_mi.shape)}")

        # b=batch size, n=节点数, f=特征维
        b, n, f = h_pl.shape
        # 如果 c 的 batch 维是 1，但节点表示是多 batch，则把 c 扩展到每个 batch。
        if c.shape[0] != b:
            if c.shape[0] == 1:
                c = c.expand(b, -1)
            else:
                raise ValueError(f"Batch mismatch: c batch={c.shape[0]} vs h_pl batch={b}")

        # 维度一致性检查：要求特征维 f 与 n_h 相同。
        if f != self.n_h or c.shape[1] != self.n_h:
            raise ValueError(
                f"Discriminator dim mismatch: expected n_h={self.n_h}, got h_pl last_dim={f}, c last_dim={c.shape[1]}."
            )

        # 设备一致性检查：避免 CPU/GPU 混用导致运行时错误。
        if h_pl.device != c.device or h_pl.device != h_mi.device:
            raise ValueError(f"Device mismatch: c={c.device}, h_pl={h_pl.device}, h_mi={h_mi.device}")

        # 只在第一次 forward 打印一次调试信息，便于排查维度/设备问题。
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(
                f"[Discriminator] n_h={self.n_h} | "
                f"h_pl={tuple(h_pl.shape)} {h_pl.dtype} {h_pl.device} | "
                f"c={tuple(c.shape)} {c.dtype} {c.device}"
            )

        # 将上下文向量 c broadcast 到每个节点：
        # c: [B,F] -> [B,1,F] -> expand 到 [B,N,F]
        # 并调用 contiguous()，避免 expand 产生的非连续内存 stride 触发底层 GEMM/CUDA 路径问题。
        # Build c_x and make tensors contiguous to avoid invalid strides in batched GEMM.
        c_x = torch.unsqueeze(c, 1).expand_as(h_pl)
        h_pl = h_pl.contiguous()
        h_mi = h_mi.contiguous()
        c_x = c_x.contiguous()

        # 输入数值检查：若上游出现 NaN/Inf，尽早抛错定位。
        # Guard: if any NaN/Inf
        if torch.isnan(h_pl).any() or torch.isnan(c_x).any():
            raise ValueError('NaN detected in discriminator inputs')
        if torch.isinf(h_pl).any() or torch.isinf(c_x).any():
            raise ValueError('Inf detected in discriminator inputs')

        # 注意：nn.Bilinear 理论上支持 (N,*,in1) 的输入，
        # 但某些 CUDA 实现对 3D + expand 的组合可能不稳定，因此这里展平成 2D。
        # Flatten to 2D then reshape back.
        h_pl2 = h_pl.view(b * n, f)
        h_mi2 = h_mi.view(b * n, f)
        c_x2 = c_x.view(b * n, f)

        # 对每个节点 i 计算双线性得分：
        # sc_1/sc_2: [B*N, 1] -> reshape 为 [B, N]
        sc_1 = self.f_k(h_pl2, c_x2).view(b, n)
        sc_2 = self.f_k(h_mi2, c_x2).view(b, n)

        # 可选 bias 加成（通常用于注入结构先验）。
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # 拼接正负得分得到 logits：维度 [B, N] + [B, N] -> [B, 2N]
        logits = torch.cat((sc_1, sc_2), 1)
        return logits

