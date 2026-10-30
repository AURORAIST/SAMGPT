# srcgraphprompt

`srcgraphprompt` 是一个基于 SAMGPT 思路改造的图表示学习与 few-shot 下游评测仓库。当前版本重点做了两件事：

1. 把下游节点评测从“单点 embedding”扩展为“k-hop 子图表示 + 原型分类”。
2. 把预训练阶段的结构/负采样逻辑改得更稳、更省显存，并补充了可解释的日志。

## 1. 仓库目标

这个版本希望解决两个问题：

- 原始 SAMGPT 的下游任务更偏向节点 embedding 直接分类，和预训练里使用的结构信息还不够一致。
- 原始实现里一些 dense 的 ego mask、负采样和 batching 方式容易导致显存压力大，或者出现退化采样。

因此这里把流程统一成：

- 预训练阶段：在源图上学习节点-子图、子图-子图的结构对齐表示。
- 下游阶段：在目标图上构造 few-shot 支持集子图，用子图 readout 生成支持/查询表示，再做原型分类。

## 2. 算法原理

### 2.1 预训练：节点-子图 + 子图-子图联合对比

对每个源图，模型先通过 `PrePrompt` / `USPPretrainingHead` 得到：

- 节点表示 `h_v`
- 以节点为中心的 ego-subgraph 表示 `g_v`

核心训练信号由三部分组成：

$$
L = L_{ns} + \lambda_{ss} L_{ss} + \lambda_{align} L_{align}
$$

其中：

- `L_ns`：node-subgraph 对比损失，鼓励节点表示和对应子图表示一致。
- `L_ss`：subgraph-subgraph 对比损失，鼓励语义/结构相近的子图更近。
- `L_align`：可选对齐项，用于结构 token 或 role bucket 的额外约束。

直观上，它不是只学“节点像什么”，而是学“节点所处局部结构像什么”。

### 2.2 下游：k-hop 子图 readout + 原型分类

对目标图的 few-shot 支持集和测试节点，先构造 k-hop ego 子图，然后用 readout 得到子图级表示：

$$
g_i = \text{Readout}(\{h_j : j \in \mathcal{N}_k(i)\})
$$

当前支持的 readout 包括：

- `mean`
- `attention`
- `prompt_weighted`

分类方式采用 prototype 风格：

1. 在支持集上按类别求类原型。
2. 用查询子图表示和各类原型做 cosine similarity。
3. 对相似度做 softmax 得到预测分布。

这比直接对单个节点 embedding 分类更一致，因为预训练和下游都显式地利用了局部结构。

## 3. 这次主要改进点

### 3.1 下游任务从 node embedding 改为 subgraph representation

原始实现里，节点 few-shot 分类更多是直接拿节点 embedding 做 prototype 分类。当前版本把 `query_mode=subgraph` 作为主路径：

- 支持集节点先扩展成子图
- 子图做 attention / prompt-weighted readout
- 再做类别原型分类

效果上，这让下游任务和预训练目标更接近，也更能利用局部结构。

### 3.2 负采样从“单图退化”改为 domain-balanced

加入了 `domain_balanced` 负采样，并显式记录：

- `num_intra_neg`
- `num_inter_neg`
- `negative_fallback_reason`

这样可以避免只在同域里采负样本，减少快捷捷径。

### 3.3 ego-subgraph 从 dense mask 改为 CSR / block cache

原来 dense ego mask 很容易 OOM。现在使用：

- CSR-style ego list
- block cache
- `max_neighbors` 截断

好处是：

- 显存占用更稳
- 大图缓存更轻
- 便于追踪每个子图的真实大小

### 3.4 batching 更可控

`source_batch_size` 不再只是“图数批大小”的概念，而是需要和图规模一起看。

经验上：

- `source_batch_size=1` 容易让 domain-balanced 负采样退化成 fallback
- `source_batch_size=2` 能恢复 inter-domain negative，但如果两个大图直接拼接，也可能顶显存

所以实际跑法要结合图大小、`num_negatives`、`ss_num_negatives` 和显存上限一起调。

### 3.5 日志更可解释

当前版本会输出：

- `max_neighbors_per_node`
- `max_subgraph_size_effective`
- `negative_fallback_reason`
- `representation_source=node_embedding / subgraph_readout`

这样可以直接看出结果到底是“真正走了子图路径”，还是退回到了节点 embedding。

## 4. 运行方式

### 4.1 环境

建议使用已有的 `graph` conda 环境：

```bash
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph
```

### 4.2 直接运行

最小 smoke test：

```bash
python srcgraphprompt/execute.py \
  --pretrain_method USP \
  --target_domain Cora \
  --source_domains Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --shots 1 \
  --seed 39 \
  --device cuda \
  --usp_epochs 1 \
  --usp_down_epochs 1 \
  --usp_eval_protocol samgpt_standard \
  --num_splits 1 \
  --split_start 0 \
  --source_batch_size 2 \
  --source_batch_node_cap 24000
```

正式评测时，可以把 `usp_epochs`、`usp_down_epochs`、`num_splits` 调回完整配置。

### 4.3 通过脚本运行

`rungraph.sh` 适合 nohup 运行并保存日志。推荐观察：

- `negative_fallback_reason` 是否为空
- `acc / macro_f1 / micro_f1`
- `representation_source=subgraph_readout`

## 5. 推荐参数

下面是当前比较稳定的一组经验参数：

- `source_batch_size=2`
- `source_batch_node_cap=24000`
- `subgraph_hop=2`
- `max_neighbors=64`
- `readout=prompt_weighted`
- `query_mode=subgraph`
- `prototype=attention`
- `negative_sampling=domain_balanced`
- `neg_exclude_scope=self`

如果显存仍然紧张，优先尝试：

1. 降低 `source_batch_node_cap`
2. 降低 `ss_num_negatives`
3. 降低 `num_negatives`
4. 临时把 `source_batch_size` 改回 1

## 6. 结果怎么解读

### 6.1 `negative_fallback_reason`

如果这个字段不是 `null`，说明 domain-balanced 负采样没有完全按理想路径工作。常见原因：

- 同 batch 里没有足够的跨域节点
- `source_batch_size` 太小
- batch 中图太大，采样被迫回退

### 6.2 `acc` 低于预期

不要直接把单次 smoke test 的结果当结论。优先检查：

- 是否只跑了 1 epoch
- 是否只跑了 1 个 split
- 是否仍在加载旧 checkpoint（`skip_pretrain=1`）
- 是否和对照组使用同样的 split / seed / 训练轮数

### 6.3 为什么要看 `macro_f1`

在类别不完全均衡时，`acc` 可能掩盖少数类问题。`macro_f1` 更能反映跨类泛化质量。

## 7. 与原始 SAMGPT 的关系

可以把这个版本理解成：

- 保留 SAMGPT 的结构 prompt / few-shot 原型范式
- 但把下游从“节点级 embedding 分类”改成“子图级表示分类”
- 同时把预训练的 ego-subgraph、负采样和日志做得更稳

所以它不是单纯的参数改动，而是对“训练目标和下游任务形式”做了对齐。

## 8. 目录说明

- `execute.py`：主训练/评测入口
- `preprompt.py`：预训练编码器与 prompt 组合
- `usp_sam.py`：USP-SAM 的对比学习与子图 readout
- `downprompt.py`：few-shot 原型分类头
- `utils/`：数据处理、ego-subgraph 生成、few-shot 划分
- `layers/`：GCN/GAT/readout/prompt 相关模块
- `models/`：基础模型组件

## 9. 备注

当前仓库里的一些实验结果文件、缓存和日志文件默认不建议提交到 git。仓库级 `.gitignore` 已经排除了常见产物。

如果你要继续写论文版说明，建议下一步补一张图：

- 预训练：node -> ego-subgraph -> contrastive
- 下游：support/query -> k-hop subgraph -> prototype classification
