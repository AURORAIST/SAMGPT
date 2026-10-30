# models 文件夹

模型定义（models）模块。包含各种图预训练方法和下游分类头的实现，是完整学习管道的核心组件。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口。

---

### `graphcl.py`
**目的**：GraphCL（Graph Contrastive Learning）预训练方法的实现。

**主要功能**：
- 对比学习框架：通过对同一图的不同增强视图进行对比，学习图的自监督表示
- 两个增强视图：都是原图的随机扰动版本（删除边/特征等）
- 判别器辅助：使用 Discriminator 进行全图-全图的对比
- 适用于多个预训练数据集的融合学习

**包含类**：
- `GraphCL(n_in, n_h, activation)` - GraphCL 预训练模型

**核心方法**：
- `forward(gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type, prompt_layers)` - 前向计算和对比损失

**使用场景**：作为基准对比学习方法，用于无监督预训练。通过 `--pretrain_method GRAPHCL` 选择。

---

### `dgi.py`
**目的**：DGI（Deep Graph Infomax）预训练方法的实现。

**主要功能**：
- 信息最大化框架：通过最大化原图与其打乱版本之间的互信息进行预训练
- 判别器辅助：区分真实编码和虚假（打乱）编码
- 比 GraphCL 更轻量级：只需一个编码器加一个判别器

**包含类**：
- `DGI(n_in, n_h, activation)` - DGI 预训练模型

**核心方法**：
- `forward(gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)` - 前向计算和 DGI 损失

**使用场景**：作为自监督预训练的替代方案，相比 GraphCL 计算更高效。

---

### `logreg.py`
**目的**：逻辑回归分类头，用于少样本(few-shot)和标准分类任务。

**主要功能**：
- 线性分类器：在预训练编码器输出之上的简单线性层
- 广泛作为其他模型的基础：原型网络(prototypical networks)等
- 支持多分类问题，使用交叉熵损失

**包含类**：
- `LogReg(ft_in, nb_classes)` - 逻辑回归分类器

**使用场景**：
- 作为独立的分类器在冻结预训练编码器时使用
- 作为更复杂下游模型的基准

---

### `mlp.py`
**目的**：多层感知机（Multi-Layer Perceptron, MLP）。

**主要功能**：
- 全连接深度网络
- 可配置的层数和隐层维度
- 用于特征变换或分类

**包含类**：
- `MLP(input_dim, hidden_dims, output_dim, activation)` 或类似

**使用场景**：作为下游任务的轻量级分类头或特征预处理模块。

---

### `gcnlayers.py`
**目的**：GCN 层级实现（低级层操作）。

**主要功能**：
- 实现单个 GCN 层的卷积操作
- 处理稀疏/稠密矩阵乘法和特征变换

**使用场景**：被 `layers/gcn.py` 调用，构成多层 GCN 编码器。

---

### `gatlayers.py`
**目的**：GAT 层级实现（低级层操作）。

**主要功能**：
- 实现单个 GAT 层的注意力计算

**使用场景**：被 `layers/gat.py` 调用，构成多层 GAT 编码器。

---

### `LP.py`
**目的**：标签传播（Label Propagation）预训练方法。

**主要功能**：
- 半监督学习框架：利用已标记和未标记节点构建预训练初始化
- 通过邻接矩阵的随机游走进行标签平滑传播
- 无需额外神经网络参数，仅基于图拓扑

**包含函数/类**：
- `LP(...)` 或相关函数 - 标签传播的实现

**使用场景**：当 `--pretrain_method LP` 或 `splitLP` 时的预训练方法。

---

## 模型使用流程

### 预训练阶段

根据 `--pretrain_method` 参数选择：
- **GraphCL 模式**：加载 `graphcl.GraphCL`，使用对比学习预训练
- **DGI 模式**：加载 `dgi.DGI`，使用信息最大化预训练
- **LP/splitLP 模式**：使用 `LP.py` 的标签传播初始化

### 下游阶段

根据任务类型选择分类头：
- **节点分类(node)**：`logreg.LogReg`（配合 GCN 编码器）
- **图分类(graph)**：`logreg.LogReg`（配合 readout 层和图级表示）
- **复杂任务**：`mlp.MLP`（可选的深层分类头）

## 配置参数（来自 execute.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrain_method` | GRAPHCL | 预训练方法：GRAPHCL、USP、LP、splitLP |
| `--backbone` | gcn | 编码器主干：gcn 或 gat |
| `--hid_units` | 256 | 隐层维度 |
| `--layers_num` | 3 | 编码器层数 |
| `--negative_samples_num` | 40 | LP/splitLP 的负样本数 |

## 技术关系

```
预训练模型选择：
  ├─ GraphCL (graphcl.py)
  │  └─ 使用 Discriminator，支持多视图对比
  │
  ├─ DGI (dgi.py)
  │  └─ 使用 Discriminator，区分真实/假样本
  │
  └─ LP / splitLP (LP.py)
     └─ 基于标签传播，无深度网络参数

下游分类头选择：
  ├─ LogReg (logreg.py) - 简单线性分类器
  └─ MLP (mlp.py) - 深度非线性分类器
```

