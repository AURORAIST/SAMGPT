# experiments 文件夹

实验运行（experiments）容器。包含正式的对比实验和消融研究的实现，用于系统评估预训练方法的性能。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口。

---

### `formal_batch1_impl.py`
**目的**：正式批次1 - 多种预训练/下游方法的系统对比实验。

**主要功能**：
- 多种子模型比较：不同预训练方法（USP、GRAPHCL等）、不同下游任务配置的全面对比
- 多分数种子(seed)运行：对同一方法配置用不同随机种子运行多次，评估结果的稳定性
- 统计度量收集：
  - 准确率(accuracy)和平衡准确率(balanced_accuracy)
  - 宏F1分数(macro_f1)和微F1分数(micro_f1)
  - 模型预测置信度间隔(margin)
  - 表示紧凑性(compactness)和类间分离度(separation)

**包含函数**：
- `_run_formal_batch1_legacy(args, parent_directory, sparse=True)` - 完整的正式实验执行流程

**使用场景**：在完整设置下进行严格的实验对比，确保结论的统计鲁棒性。

---

### `query_objective_ablation_impl.py`
**目的**：查询模式和目标函数(objective)消融研究。

**主要功能**：
- 查询模式消融(query mode ablation)：
  - `node` - 仅使用单节点作为查询
  - `subgraph` - 使用k-hop子图作为查询
  - `hybrid` - 混合节点和子图查询
  
- 目标函数权重消融：
  - `lambda_ns` - 节点-子图对比学习权重
  - `lambda_ss` - 子图-子图对比学习权重
  - `lambda_align` - 对齐损失权重
  
- 参数梯度追踪：记录各层（编码器、读出、prompt）的参数更新幅度(delta norms)

**包含函数**：
- `_run_query_objective_ablation_legacy(args, parent_directory, sparse=True)` - 完整的消融实验执行流程

**使用场景**：深入分析模型各个设计选择（查询策略、损失项权重）对最终性能的影响，指导模型结构和超参优化。

---

## 数据流

```
execute.py
  ↓ 根据 --formal_batch1 / --query_objective_ablation 标志调用对应的实现
experiments/*_impl.py
  ↓ 导入共享工具、模型构建函数
utils.usp_shared.*、preprompt、downprompt 等
```

## 配置参数（来自 execute.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--formal_batch1` | False | 是否运行正式批次1实验 |
| `--query_objective_ablation` | False | 是否运行查询/目标消融实验 |
| `--formal_seeds` | [0,1,2] | 正式实验的随机种子列表 |
| `--formal_epoch` | 50 | 正式实验中预训练的轮数 |
| `--query_mode` | hybrid | 查询模式：node、subgraph、hybrid |
| `--usp_lambda_ns` | 1.0 | 节点-子图损失权重 |
| `--usp_lambda_ss` | 1.0 | 子图-子图损失权重 |
| `--usp_lambda_align` | None | 对齐损失权重（默认等于 --alpha） |

## 输出

- **JSON 日志文件**：记录详细的实验过程和结果指标
- **CSV 结果文件**：`result/<set_name>.csv`，汇总各方法的性能对比
- **标准输出**：打印实验进度和阶段性成果

