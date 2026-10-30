# SAMGPT GraphCL 实验记录 2026-05-08

## 目标

本轮尝试改进跨域 GraphCL 预训练：

1. 将 GraphCL 改成 mini-batch 节点训练。
2. 在 mini-batch 中使用其他节点作为负样本，补充节点级 InfoNCE。
3. 引入聚类筛选/聚类质量权重，让高质量节点更强地参与预训练。
4. 避免跨域数据规模和聚类难度不同导致某些域贡献过多、某些域贡献过少。

## 已做代码修改

### 1. Mini-batch GraphCL

相关文件：

- `src/models/graphcl.py`
- `src/preprompt.py`
- `src/execute.py`

GraphCL 支持传入：

- `graphcl_node_indices`: 当前 batch 的 anchor 节点。
- `graphcl_negative_indices`: 当前 batch 的负样本节点。

这样仍然在全图上传播 GNN，但只在选中的节点上计算对比损失，减少显存压力。

### 2. 节点级 in-batch InfoNCE

相关函数：

- `PrePrompt.compute_graphcl_node_embeddings`
- `PrePrompt.graphcl_inbatch_nce_loss`

损失形式：

- 同一节点在两种增强视图下的 embedding 为正样本。
- batch 中其他节点为负样本。
- 非 streaming 模式下，不同域的节点会拼到同一个 batch，因此负样本包含跨域节点。
- streaming 模式下，每次只处理一个域，因此 NCE 负样本只来自当前域。

新增参数：

```bash
--graphcl_nce_lambda
--graphcl_nce_temperature
```

### 3. 聚类质量 soft weight

相关函数：

- `compute_cluster_quality_weights`

标准：

- 每个域内部单独聚类。
- 节点到簇中心越近，质量权重越高。
- 每个域内权重归一化到均值 1，避免某个域因为更容易聚类或特征尺度不同而天然占优。

新增参数：

```bash
--cluster_quality_weight
```

### 4. 修复 mini-batch 每轮重复采样问题

原训练循环每个 epoch 都执行：

```python
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
```

这会导致每个 epoch 抽到相同的 mini-batch 和负样本。已改成：

```python
np.random.seed(seed + epoch)
torch.manual_seed(seed + epoch)
torch.cuda.manual_seed(seed + epoch)
```

## 本轮实验设置

本轮运行的核心设置：

```bash
--graphcl_batch_size 1024
--graphcl_nce_lambda 0.2
--graphcl_nce_temperature 0.2
--cluster_quality_weight 1
--domain_align_lambda 0.0
```

说明：

- 没有开启 domain prototype alignment，因为之前 `domain_align_lambda=0.1` 的结果明显负迁移。
- 使用节点级 NCE，权重为 0.2。
- 开启聚类质量 soft weight。

## 关键日志

训练中期：

```text
epoch=142 Loss:[0.74469656] graphcl=0.01649000 align=0.00000000 nce=3.64103246
epoch=143 Loss:[0.74428058] graphcl=0.01500737 align=0.00000000 nce=3.64636612
epoch=144 Loss:[0.74350309] graphcl=0.01392981 align=0.00000000 nce=3.64786625
epoch=145 Loss:[0.74138582] graphcl=0.01419360 align=0.00000000 nce=3.63596106
```

epoch 999：

```text
epoch=999 Loss:[0.62661988] graphcl=0.00202157 align=0.00000000 nce=3.12299132
```

validation：

```text
pretrain_epoch_1000 val Average accuracy:[27.7200]
Mean:[27.7200]
Std :[5.8708]
```

## 结果判断

本轮是明显负迁移。

原因判断：

1. 总损失几乎完全由 NCE 主导。

   ```text
   Loss ≈ graphcl + 0.2 * nce
   0.6266 ≈ 0.0020 + 0.2 * 3.1230
   ```

2. 原始 GraphCL BCE loss 很快降到接近 0，说明该任务对模型来说过于容易。

3. NCE 长期保持在 3 左右，且下游验证只有 27.72，说明当前 NCE 约束过强，可能在过度推远节点。

4. 跨域场景下，“其他节点都是负样本”的假设风险很高。不同域中可能存在语义相似或对下游有用的节点结构，被 NCE 当成负样本强行推远。

5. 聚类质量 soft weight 可能也会放大错误筛选，因此需要单独消融。

## 与已有结果对比

已有 CSV 结果显示：

```text
baseline / full nodes: 约 45.59 - 46.40
cluster_keep_ratio=0.5: 42.81
domain_align_lambda=0.1: 25.33
本轮 NCE 0.2 + cluster weight: val 27.72
```

结论：

- 强 domain alignment 会严重负迁移。
- 强 NCE 也会严重负迁移。
- 硬聚类筛选 `keep_ratio=0.5` 已经有负收益。
- 需要先确认“修复每轮采样”本身是否能恢复或提升 baseline，再逐步加 NCE。

## 下一步消融建议

优先跑三组：

### A. 只修复每轮采样，不加 NCE

```bash
--graphcl_nce_lambda 0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

目的：

- 验证 mini-batch 每轮变化是否本身有帮助。
- 作为新代码 baseline。

### B. 小权重 NCE

```bash
--graphcl_nce_lambda 0.05
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

目的：

- 检查 NCE 是否在较小权重下能提供正收益。
- 先关闭聚类权重，避免多个因素混在一起。

### C. 小权重 NCE + 聚类 soft weight

```bash
--graphcl_nce_lambda 0.05
--cluster_quality_weight 1
--domain_align_lambda 0.0
```

目的：

- 检查聚类质量权重是否有帮助。

## 当前建议

不要继续使用：

```bash
--graphcl_nce_lambda 0.2
```

该权重过强，当前结果已经显示明显负迁移。

短期推荐：

---

## 2026-05-09 追加：默认 NCE + 动态去噪（跨域）实验

### 本轮新增实现

1. `graphcl_loss_mode` 默认改为 `nce`（单损失，不再默认叠加 BCE）。
2. 加入动态去噪：
   - 基于两视图正样本相似度更新节点 EMA 可靠性。
   - 基于当前 batch embedding 相似度屏蔽高相似负样本。
3. 修复跨域 mask 传递：
   - 非 streaming 下支持全局 `global_negative_mask`，避免被块对角 mask 隐式退化为域内负样本。
4. 日志增强：
   - `dynamic-denoise epoch=... pos_sim_mean dyn_weight_mean drop_ratio`
   - `NCE-DIAG ... likely_fn_ratio high_sim_neg ...`

### 本轮关键结果（20260509.log）

- `pretrain_epoch_1000 val Mean = 33.33, Std = 7.5760`
- `pretrain_epoch_2000 val Mean = 31.87, Std = 7.1896`
- best checkpoint 出现在 `epoch=1000`（后续继续训练出现回落）。

训练诊断（关键）：

- `dynamic drop_ratio` 约 `0.007 ~ 0.0085`（仅过滤约 0.7%~0.85% 负样本）
- `likely_fn_ratio` 长期约 `0.194 ~ 0.198`（疑似假负样本占比仍高）
- 静态分支 `safe_ratio=1.0`，几乎不提供额外过滤。

### 判断

1. 当前“默认 NCE + 动态去噪”是有效改进（相对之前 27.72 提升到 33.33）。
2. 但还不够好，且 1000 轮后有过训/收益回落。
3. 核心瓶颈是去噪力度不足：动态 mask 只移除极少量负样本，无法显著降低假负样本污染。

### 下一步（只做一处改动）

优先只改动态阈值，增强过滤强度，其他参数固定：

1. `--dynamic_denoise_sim_threshold 0.55`
2. `--dynamic_denoise_sim_threshold 0.50`
3. `--dynamic_denoise_sim_threshold 0.45`

目标：

- 把 `drop_ratio` 提升到 `2% ~ 8%`
- 观察 `likely_fn_ratio` 是否下降
- 比较 `val@1000` 是否超过 `33.33`

如果三组都不提升，再进行下一刀（单改）：

- `--graphcl_nce_lambda` 从 `0.5` 降到 `0.2`

### 建议执行策略

- 不再盲目拉长 epoch；统一以 `epoch=1000` 为比较点。
- 每组只跑到 1000 并记录：
  - `val mean/std`
  - `drop_ratio`
  - `likely_fn_ratio`
  - `best epoch`

```bash
--graphcl_nce_lambda 0.05
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

等 A/B/C 三组结果出来后，再决定是否重新设计跨域负样本策略，例如：

- 只把低相似节点作为负样本。
- 使用跨域 prototype 做软负样本，而不是所有其他节点都当负样本。
- 对 NCE 加 warmup，避免训练早期过强推远。
- 先域内 NCE，再逐步加入跨域 NCE。

## 追加实验：NCE 0.05，关闭聚类权重

运行设置：

```bash
--graphcl_batch_size 1024
--graphcl_nce_lambda 0.05
--graphcl_nce_temperature 0.2
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

epoch 999 训练日志：

```text
epoch=999 Loss:[0.16973256] graphcl=0.00017047 align=0.00000000 nce=3.39124179
```

第一次 validation：

```text
pretrain_epoch_1000 val Average accuracy:[30.2400]
Mean:[30.2400]
Std :[6.7106]
```

判断：

- 相比 `graphcl_nce_lambda=0.2` 的 `27.72`，`0.05` 有改善。
- 但仍显著低于已有 baseline 的约 `45.59 - 46.40`。
- 即使关闭聚类权重，NCE 仍然明显负迁移。
- 当前最可能的问题不是聚类 soft weight，而是“其他节点都作为负样本”的假设过强，破坏了跨域可迁移结构。

下一轮建议：

```bash
--graphcl_nce_lambda 0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

目的：

- 只验证 mini-batch 化和 epoch-varying 采样本身。
- 如果该设置能回到 45-46 附近，说明新增 NCE 是主要负迁移来源。
- 如果仍然很低，则需要检查 mini-batch GraphCL 的 BCE 形式本身是否改变了原方法的优化行为。

## 追加实验：NCE 0，仅保留 mini-batch GraphCL

运行设置：

```bash
--graphcl_batch_size 1024
--graphcl_nce_lambda 0
--graphcl_nce_temperature 0.2
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

epoch 1000 第一次 validation：

```text
pretrain_epoch_1000 val Average accuracy:[44.0000]
Mean:[44.0000]
Std :[11.0271]
```

判断：

- 该结果显著高于 `NCE=0.2` 的 `27.72` 和 `NCE=0.05` 的 `30.24`。
- 因此当前主要负迁移来源基本可以确定为节点级 in-batch NCE，而不是 mini-batch 化本身。
- 但 `44.00` 仍低于已有 full/all-node 结果约 `45.59 - 46.40`，说明 `batch_size=1024` 的 mini-batch GraphCL 还没有带来正收益。
- 标准差 `11.03` 较高，第一次 validation 不一定足够稳定，可以继续观察 epoch 2000/3000 的 downstream validation。

下一步建议：

1. 先继续当前 `NCE=0` 训练，观察 epoch 2000 validation 是否超过 45-46。
2. 如果 epoch 2000 仍低于 baseline，下一轮建议扩大 batch：

```bash
--graphcl_batch_size 4096
--graphcl_nce_lambda 0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

理由：

- `1024` 可能引入过多采样噪声。
- 不加 NCE 时，batch 越大越接近原 GraphCL 全节点训练，同时仍保留小批量机制。

3. 暂时不要继续尝试普通 in-batch NCE。若后续重新设计 NCE，应改成软负样本或 hard negative 筛选，而不是“所有其他节点都是负样本”。

epoch 2000 第二次 validation：

```text
pretrain_epoch_2000 val Average accuracy:[44.5400]
Mean:[44.5400]
Std :[11.2478]
```

补充判断：

- 相比 epoch 1000 的 `44.00`，epoch 2000 提升到 `44.54`，说明继续训练有小幅收益。
- 但仍低于已有 full/all-node baseline 的约 `45.59 - 46.40`，当前 `batch_size=1024, NCE=0` 还没有形成正收益。
- 目前实验链条比较清楚：`NCE=0.2 -> 27.72`，`NCE=0.05 -> 30.24`，`NCE=0 -> 44.54`。普通节点级 in-batch NCE 是当前最主要的负迁移来源。

下一轮更建议优先跑大 batch 的纯 mini-batch GraphCL：

```bash
--graphcl_batch_size 4096
--graphcl_nce_lambda 0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

目的：

- 判断 `1024` batch 的采样噪声是否造成了低于 baseline。
- 如果 `4096` 能接近或超过 `45.6`，再在这个更稳的基础上设计跨域负样本。
- 如果 `4096` 仍低于 baseline，则 mini-batch GraphCL 本身可能需要加入 memory bank 或 batch-balanced domain sampler，而不是简单扩大 batch。

epoch 3000 第三次 validation：

```text
pretrain_epoch_3000 val Average accuracy:[43.7900]
Mean:[43.7900]
Std :[11.2522]
```

补充判断：

- epoch 3000 从 epoch 2000 的 `44.54` 回落到 `43.79`。
- 当前这条 `batch_size=1024, NCE=0` 的最好验证点是 epoch 2000，best=`44.54`。
- 继续训练没有看到持续提升，说明该设置基本进入平台/波动区间。
- 下一轮优先看 `batch_size=4096, NCE=0` 是否能降低 mini-batch 采样噪声并接近 full/all-node baseline。

## 追加实验：NCE 0，扩大 batch 到 4096

运行设置：

```bash
--graphcl_batch_size 4096
--graphcl_nce_lambda 0
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--domain_align_lambda 0.0
```

epoch 4000 validation：

```text
pretrain_epoch_4000 val Average accuracy:[43.7000]
Mean:[43.7000]
Std :[10.7717]
```

判断：

- 该设置没有超过 `batch_size=1024, NCE=0` 的最好结果。
- 扩大 batch 没有解决性能低于 full/all-node baseline 的问题。
- 下一步不继续单纯扩大 batch，转向“对比去噪”：先只加 anchor soft weighting，不加 NCE，验证聚类去噪是否至少不会造成普通 NCE 那样的负迁移。

## 追加实验：聚类去噪 anchor weighting，不加 NCE

运行设置：

```bash
--graphcl_batch_size 4096
--graphcl_nce_lambda 0
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--denoise_anchor_weight 1
--denoise_negative_mask 0
--denoise_weight_alpha 1.0
--denoise_visualize 1
--domain_align_lambda 0.0
```

验证结果：

```text
pretrain_epoch_1000 val Average accuracy:[44.1800]
pretrain_epoch_2000 val Average accuracy:[44.3400]
pretrain_epoch_3000 val Average accuracy:[44.0300]
pretrain_epoch_4000 val Average accuracy:[43.4600]
pretrain_epoch_5000 val Average accuracy:[43.4200]
best validation: epoch 2000, 44.3400

final test Average accuracy:[46.6300]
Mean:[46.6300]
Std :[12.4102]
```

可视化输出：

```text
result/denoise_visualizations/model_node_GRAPHCL__Citeseer_Pubmed_Photo_Computers_FacebookPagePage_LastFMAsia_1.0_1.0_all_all_50_256_0.001_gcn_gclbs4096_gclneg1_gclnce0.0_gclstream1_clkeep1.0_clnum10_clbalfixed_clquota0_denaw1_denmask0_denth0.6_dalign0.0_dcl0/
```

判断：

- 这组验证最好值 `44.34`，略低于 `batch_size=1024, NCE=0` 的 `44.54`，但 final test 达到 `46.63`。
- 相比已有 full/all-node baseline 约 `45.59 - 46.40`，final test 有小幅正收益。
- 该结果说明“去噪 anchor weighting”至少不是明显负迁移，并且比普通 NCE 稳定得多。
- 当前更合理的解释是：聚类分数不适合直接做硬筛选或强对齐，但可以作为弱可靠性权重，降低可疑 anchor 对 GraphCL BCE 的贡献。

## 追加实验：聚类去噪 anchor weighting + masked NCE

运行设置：

```bash
--graphcl_batch_size 2048
--graphcl_nce_lambda 0.005
--graphcl_nce_temperature 0.2
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--denoise_anchor_weight 1
--denoise_negative_mask 1
--denoise_sim_threshold 0.6
--denoise_min_negatives 16
--denoise_weight_alpha 1.0
--denoise_visualize 1
--domain_align_lambda 0.0
```

验证结果：

```text
pretrain_epoch_1000 val Average accuracy:[41.7500]
pretrain_epoch_2000 val Average accuracy:[36.8000]
pretrain_epoch_3000 val Average accuracy:[33.7200]
pretrain_epoch_4000 val Average accuracy:[32.6900]
best validation: epoch 1000, 41.7500

final test Average accuracy:[45.2400]
Mean:[45.2400]
Std :[12.4000]
```

mask 统计：

```text
safe_ratio ~= 0.80 - 0.81
safe_neg_mean ~= 1640 - 1657 / 2047
```

可视化输出：

```text
result/denoise_visualizations/model_node_GRAPHCL__Citeseer_Pubmed_Photo_Computers_FacebookPagePage_LastFMAsia_1.0_1.0_all_all_50_256_0.001_gcn_gclbs2048_gclneg1_gclnce0.005_gclstream1_clkeep1.0_clnum10_clbalfixed_clquota0_denaw1_denmask1_denth0.6_dalign0.0_dcl0/
```

判断：

- masked NCE 比之前的裸 NCE 明显温和，但仍然导致验证性能随训练持续下降。
- 当前 mask 只过滤约 20% 负样本，剩余负样本里仍可能包含跨域语义相近但结构/特征不一致的 false negative。
- 即使 NCE 权重只有 `0.005`，它仍然会给表示空间加入“推远其他节点”的方向，和跨域可迁移性目标冲突。
- 因此现阶段不建议继续把 NCE 作为主损失加进去。

## 当前结论 2026-05-09

目前已经找到一个关键原因：

- 负收益主要来自普通节点级 NCE 的 false negative 问题。
- “其他样本都作为负样本”在跨域图上不成立；即使用聚类和相似度 mask 过滤一部分，仍然会留下足够多的错误负样本。
- 聚类信息更适合先作为去噪权重，而不是作为硬筛选、强对齐或 NCE 负样本依据。

当前最值得保留的设置：

```bash
--graphcl_batch_size 4096
--graphcl_nce_lambda 0
--cluster_keep_ratio 1.0
--cluster_quality_weight 0
--denoise_anchor_weight 1
--denoise_negative_mask 0
--denoise_weight_alpha 1.0
--domain_align_lambda 0.0
```

下一步建议：

1. 先围绕 anchor-only denoising 做小范围消融：

```bash
--denoise_weight_alpha 0.5
--denoise_weight_alpha 1.5
```

2. 暂时不要继续普通 NCE 或 masked NCE。

3. 如果还要尝试 NCE，应改成更保守的方案，例如 warmup 之后再开、`lambda=0.001`、或者只把低相似且不同簇的样本作为非常弱的负样本。

4. 需要结合可视化检查每个域的 reliability 分布，重点看是否某些域的高权重节点过于集中。如果集中严重，下一版需要加 domain-balanced reliability normalization。
