import torch
import torch.nn as nn


class textprompt(nn.Module):
    """最基础的“文本提示(prompt)”模块。

    这里的 prompt 本质上是一个可学习向量 `weight`（形状 [1, hid_units]），
    用于对输入 embedding 做：
    - add: 逐样本相加
    - mul: 逐元素相乘（默认）

    典型用法：把图/节点 embedding 按照 prompt 进行调制（modulation）。
    """

    def __init__(self, hid_units, type_='mul'):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        # 可学习 prompt 向量
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 初始化，适合线性层/向量参数
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        """对输入 embedding 应用 prompt。

        Args:
            graph_embedding: 形状通常为 [B, hid_units]（也可以是可广播匹配的形状）

        Returns:
            应用 add/mul 后的 embedding（形状与输入一致）
        """
        if self.prompttype == 'add':
            # 将 [1,F] repeat 到 [B,F] 以匹配 batch
            weight = self.weight.repeat(graph_embedding.shape[0], 1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            # 逐元素缩放（broadcast）
            graph_embedding = self.weight * graph_embedding

        return graph_embedding


class weighted_prompt(nn.Module):
    """对多个 embedding 做可学习加权融合。

    - 内部参数 `weight`: [1, weightednum]，每个分量对应一个输入 embedding 的系数。
    - forward 期望 `graph_embedding` 是一个“长度为 weightednum 的列表/序列”，其中每个元素是同形状张量。

    注意：当前实现不会对权重做 softmax/归一化，权重可以任意取值。
    """

    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # 直接均匀初始化到 [0,1)
        self.weight.data.uniform_(0, 1)

    def forward(self, graph_embedding):
        # graph_embedding: List[Tensor]，长度必须等于 weightednum
        # print("weight",self.weight)
        # graph_embedding=torch.mm(self.weight, graph_embedding)
        assert len(graph_embedding) == self.weight.shape[1], 'length must equal'

        # 用第一个元素的形状创建累加器
        ans = torch.zeros_like(graph_embedding[0])
        for i in range(len(graph_embedding)):
            ans += self.weight[0][i] * graph_embedding[i]
        return ans


class combineprompt(nn.Module):
    """将两个 embedding 通过可学习标量权重线性组合后再做激活。"""

    def __init__(self):
        super(combineprompt, self).__init__()
        # 两个系数，对应 graph_embedding1 与 graph_embedding2
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding1, graph_embedding2):
        # 线性组合
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        # 再过非线性
        return self.act(graph_embedding)


class composedtoken(nn.Module):
    """把多个“token 向量”组合成一个 token，并用于调制输入序列。

    初始化时：
    - `texttokens`：token 列表（每个张量形状通常是 [1,F] 或 [T_i,F]，要求 dim=0 可拼接）
    - 先在 dim=0 上 cat 成一个大的 `self.texttoken`
    - 再用 `weighted_prompt` 学习如何把这些 token 加权融合成最终 token

    forward 时：
    - 得到融合后的 texttoken
    - 对输入 seq 做 add 或 mul（与 `textprompt` 类似）
    """

    def __init__(self, texttokens, type_='mul'):
        super(composedtoken, self).__init__()
        # print(texttoken1.shape)
        self.texttoken = torch.cat(texttokens, dim=0)
        # print(self.texttoken.shape)
        self.prompt = weighted_prompt(len(texttokens))
        self.type = type_

    def forward(self, seq):
        # print(seq.shape)
        # 先对 token 列表做加权融合
        texttoken = self.prompt(self.texttoken)

        # print(texttoken.shape)
        if self.type == 'add':
            # 将 token repeat 到 batch
            texttoken = texttoken.repeat(seq.shape[0], 1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets


class composedNet(nn.Module):
    """对多组参数字典（state_dict-like）做逐 key 的可学习加权融合。

    用途示例：
    - 将多个模型/多个 adapter 的参数按可学习权重组合成一个“合成”参数集合。

    注意：
    - 这里在 __init__ 里直接对 prompt 调用了 `.cuda()`，会强制把参数放到 GPU。
      如果在 CPU 环境或使用非默认 device，可能会产生 device mismatch。
    - forward 中假设 `paras` 是 list[dict]，且每个 dict 有相同的 key 集合，value 同形状。
    """

    def __init__(self, length):
        super(composedNet, self).__init__()
        # self.texttoken = torch.cat(texttokens,dim=0)
        self.length = length
        self.prompt = weighted_prompt(length).cuda()

    def forward(self, paras):
        # print(seq.shape)
        assert self.length == len(paras), 'number of paras must equal to self.length'

        # 先用第一个参数字典的结构初始化 target
        target = {}
        for key, value in paras[0].items():
            target[key] = torch.zeros_like(value)

        # 对每个 key 收集所有 paras[i][key]，再用 weighted_prompt 融合
        for key in paras[0].keys():
            para_key = [para[key] for para in paras]
            target[key] = self.prompt(para_key)

        return target
