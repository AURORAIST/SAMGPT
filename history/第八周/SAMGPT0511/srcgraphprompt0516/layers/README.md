# layers 文件夹

神经网络层（network layers）模块。包含 GNN 编码器、注意力层、读出层、提示层等基础组件，是模型的构建砖块。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口（通常导出 GCN、GAT、AvgReadout 等常用类）。

---

### `gcn.py`
**目的**：图卷积网络（Graph Convolutional Network, GCN）的实现。

**主要功能**：
- 多层 GCN 编码器，逐层传播节点特征并聚合邻接信息
- 支持稀疏(sparse)和稠密(dense)的邻接矩阵格式
- 多层堆叠和非线性激活函数配置
- 用于对图中的节点进行特征编码和表示学习

**包含类**：
- `GCN` - 多层图卷积编码器

**使用场景**：作为预训练模型和下游模型的主干GNN编码器，将图的拓扑和特征信息编码成节点表示。

---

### `gat.py`
**目的**：图注意力网络（Graph Attention Networks, GAT）的实现。

**主要功能**：
- 基于图注意力机制的编码器
- 每个节点学习对邻接节点不同的注意力权重
- 相比GCN更灵活的邻接聚合方式
- 支持多头注意力（multi-head attention）

**包含类**：
- `GAT` - 多层图注意力编码器

**使用场景**：作为 GCN 的替代编码器，特别是在需要动态调整邻接影响权重时。可通过 `--backbone` 参数选择使用。

---

### `gcnlayers.py`
**目的**：GCN 的基础层级实现（单层 GCN 操作）。

**主要功能**：
- 实现单个 GCN 层的前向计算（稀疏和稠密矩阵乘法）
- 被 `gcn.py` 中的多层 GCN 调用
- 处理特征变换、邻接传播等低级操作

**包含类**：
- `GCNLayer` 或相似的单层实现

---

### `gatlayers.py`
**目的**：GAT 的基础层级实现（单层 GAT 操作）。

**主要功能**：
- 实现单个 GAT 层的注意力计算和聚合
- 被 `gat.py` 中的多层 GAT 调用

**包含类**：
- `GATLayer` 或相似的单层实现

---

### `discriminator.py`
**目的**：对比学习中的判别器（discriminator）。

**主要功能**：
- DGI (Deep Graph Infomax) 和 GraphCL 预训练方法中的二分类判别器
- 用于区分真实的图表示和随机打乱的表示（或增强视图）
- 与编码器协作进行对比学习

**包含类**：
- `Discriminator` - 标准判别器

**使用场景**：当使用 GRAPHCL 或 DGI 预训练方法时，用于辅助对比目标的学习。

---

### `discriminator2.py`
**目的**：改进或备用的判别器实现。

**主要功能**：
- 可能是 discriminator.py 的增强版本或不同的变体
- 用于特定预训练场景

**包含类**：
- `Discriminator2` - 备用判别器

**使用场景**：在需要不同判别器设计时替换标准 Discriminator。

---

### `readout.py`
**目的**：图级/子图级读出层（graph-level readout）。

**主要功能**：
- 将节点级表示聚合成图级表示（graph-level embedding）
- 常见的聚合方式：
  - 平均池化（mean pooling / AvgReadout）
  - 注意力加权池化（attention-based readout）
  - 提示加权池化（prompt-weighted readout）
- 用于 few-shot 学习中计算支撑集的原型表示

**包含类**：
- `AvgReadout` - 平均读出
- `AttentionReadout` - 注意力读出（可能）
- `PromptWeightedReadout` - 提示加权读出（可能）

**使用场景**：在下游 few-shot 分类中，将支撑集节点/子图的 GNN 输出聚合为类原型表示。

---

### `prompt.py`
**目的**：可学习提示层（prompt layers），融合不同数据源或任务信息。

**主要功能**：
- **文本提示(textprompt)**：可学习向量，对输入表示做逐元素乘法或加法调制
- **多源融合**：结合不同预训练数据集或不同任务的信息
- **结构化提示**：可能包含针对图拓扑特征的特殊处理

**包含类**：
- `textprompt` - 基础文本/融合提示层

**使用场景**：
- 预训练阶段：融合多个源数据集的表示
- 下游微调：通过提示层适应新任务，同时保留预训练知识

---

## 架构关系

```
模型构建流程：
  ├─ 编码器选择
  │  ├─ GCN (gcn.py) / GAT (gat.py)
  │  └─ 基础层 (gcnlayers.py / gatlayers.py)
  │
  ├─ 判别器（对比预训练时使用）
  │  └─ Discriminator / Discriminator2 (discriminator*.py)
  │
  ├─ 读出层（图级表示）
  │  └─ AvgReadout / AttentionReadout (readout.py)
  │
  └─ 提示层（融合）
     └─ textprompt (prompt.py)
```

所有这些层在 `preprompt.py` 中被组装成完整的预训练模型。

