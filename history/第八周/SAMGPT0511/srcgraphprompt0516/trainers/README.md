# trainers 文件夹

训练器（trainers）模块。包含完整的预训练实现，特别是 USP-SAM（Unsupervised Subgraph Pooling with Structural Alignment Modulation）方法的预训练和下游评测管道。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口。

---

### `usp_minimal.py`
**目的**：USP-SAM 预训练方法的动态加载入口（已弃用或备用）。

**主要功能**：
- 可能是为了动态加载第三方 USP-SAM 库或保留的兼容性代码
- 现在的实现通常在其他模块（如 usp_minimal_impl.py）中

**使用场景**：备用接口，主要实现转移到 `usp_minimal_impl.py`。

---

### `usp_minimal_impl.py`
**目的**：USP-SAM 完整预训练和下游评测实现的核心代码。

**主要功能**：

#### 1. 预训练阶段（upstream）
- **多数据源融合**：同时在多个预训练图数据集（Citeseer, Pubmed, Photo 等）上进行预训练
- **子图采样和缓存**：利用 `usp_sam.ego_subgraph_mask` 提取 k-hop ego-subgraph，并缓存 CSR 格式以加速
- **批量策略**：支持源图批处理（`--source_batch_size`），在一个优化器步骤中处理多个源图
- **负采样**：两种模式
  - `domain_balanced` - 域内和域间负样本均衡
  - `random` - 随机负采样
- **对比目标**：
  - L_ns：节点-子图对比学习（node-subgraph contrastive loss）
  - L_ss：子图-子图对比学习（subgraph-subgraph contrastive loss）
  - L_align：可选对齐损失
- **损失聚合**：
  - `--usp_lambda_ns` - 节点-子图损失权重（默认 1.0）
  - `--usp_lambda_ss` - 子图-子图损失权重（默认 1.0）
  - `--usp_lambda_align` - 对齐损失权重（默认等于 --alpha）

#### 2. 下游评测阶段（downstream）
- **目标图特征和邻接加载**：使用与预训练相同的 PCA 压缩维度
- **Few-shot 分割**：从预先生成的分割文件读取支撑集和查询集
- **子图构建**：基于 `--query_mode` 和 `--subgraph_hop` 构建查询样本的子图
- **下游头训练**：在冻结预训练编码器的情况下，训练 downprompt 分类头
- **评测协议**：
  - `single_split` - 单个分割评测（快速测试）
  - `samgpt_standard` - 标准 SAMGPT 协议（100 个分割，每个分割多个 shot）
- **指标记录**：准确率、宏/微 F1 分数、边界间隔(margin)、紧凑性(compactness)、分离度(separation)

**包含函数**：
- `_run_usp_minimal_legacy(args, parent_directory, current_dir, sparse=True)` - 完整的 USP-SAM 预训练和下游评测流程

**超参可配置项**：
- `--usp_epochs` - 预训练轮数（默认 20）
- `--usp_down_epochs` - 下游微调轮数（默认 100）
- `--subgraph_hop` - ego-subgraph 的跳数(k-hop)，默认 2
- `--max_neighbors` - 每个节点的最大邻接数，默认 64
- `--query_mode` - 查询模式：node、subgraph、hybrid
- `--readout` - 子图读出方式：mean、attention、prompt_weighted
- `--prototype` - 原型聚合方式：mean、attention
- `--temperature` - InfoNCE 温度系数，默认 0.2
- `--use_structure_token` - 是否使用结构提示层
- `--neg_exclude_scope` - 负采样排除范围：ego、self、none

**性能特性**：
- 缓存友好：在 CSR 格式下高效存储子图邻接，避免重复计算
- 内存高效：块级缓存(block caching)可规模到大型图
- 可扩展：支持多GPU并行（通过外层框架）

**使用场景**：
- 主要的 USP-SAM 预训练实验运行
- 当 `--pretrain_method USP` 时由 execute.py 调用
- 完整的端到端工作流：预训练 → 下游 few-shot 评测

---

## 数据流和依赖

```
execute.py
  ↓ 调用 run_usp_minimal（当 --pretrain_method USP 时）
trainers/usp_minimal_impl.py
  ├─ 导入共享工具
  │  └─ utils.usp_shared（_usp_load_graph, _usp_device, build_or_load_ego_cache 等）
  │
  ├─ 导入核心库
  │  ├─ usp_sam（ego_subgraph_mask, _mask_to_csr）
  │  ├─ preprompt.PrePrompt（预训练主模型）
  │  ├─ downprompt（下游分类头）
  │  └─ torch.cuda.amp.GradScaler（混合精度训练）
  │
  └─ 输出结果
     ├─ checkpoints/<set_name>.pkl（最优预训练权重）
     └─ result/<set_name>.csv（下游评测指标）
```

## 配置参数汇总

### 预训练超参
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--usp_epochs` | 20 | 预训练总轮数 |
| `--pretrain_datasets` | [Citeseer, Pubmed, Photo, ...] | 预训练数据集列表 |
| `--source_batch_size` | 0 | 源图批大小（0 禁用批处理） |
| `--negative_sampling` | domain_balanced | 负采样模式 |
| `--num_negatives` | 128 | 每个锚点的负样本数 |
| `--intra_neg_ratio` | 0.5 | 域内/域间负样本比例 |

### 子图构建超参
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subgraph_hop` | 2 | K-hop ego-subgraph 范围 |
| `--max_neighbors` | 64 | 每节点最大邻接数 |
| `--csr_readout` | True | 使用 CSR 格式存储子图 |

### 损失与目标超参
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--usp_lambda_ns` | 1.0 | 节点-子图损失权重 |
| `--usp_lambda_ss` | 1.0 | 子图-子图损失权重 |
| `--usp_lambda_align` | None | 对齐损失权重 |
| `--usp_temperature` | 0.2 | InfoNCE 温度 |

### 下游超参
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--usp_down_epochs` | 100 | 下游微调轮数 |
| `--usp_eval_protocol` | single_split | 评测协议：single_split 或 samgpt_standard |
| `--num_splits` | 100 | 评测分割总数 |
| `--shots` | None | Few-shot 元组（默认 [shot_num]） |

## 输出文件

- **Checkpoint**：`checkpoints/<set_name>.pkl` - 最优预训练模型权重
- **结果 CSV**：`result/<set_name>.csv` - 各 shot 和 split 的详细性能评测
- **日志和诊断**：
  - `log.txt` - 完整的标准输出日志
  - 可选的时间профиль CSV（`--time_profile True`）

