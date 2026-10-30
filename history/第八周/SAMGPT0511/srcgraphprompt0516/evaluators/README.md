# evaluators 文件夹

下游评估（downstream evaluation）模块。用于实现图预训练模型在下游任务上的性能评测和校准实验。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口。

### `sanity_calibration_impl.py`
**目的**：实现基础版本的 sanity calibration 实验。

**主要功能**：
- 日志打乱（label shuffle）：将目标图的标签随机重排，测试模型对误导标签是否过拟合
- 边扰动（edge perturbation）：随机删除图的边，检测模型对图拓扑的依赖程度
- 特征扰动（feature dropout）：随机删除或掩蔽节点特征，测试模型对特征的鲁棒性
- 评测指标：准确率(accuracy)、宏F1分数(macro F1)、紧凑性(compactness)、分离度(separation)

**包含函数**：
- `_run_sanity_calibration_legacy(args, parent_directory, sparse=True)` - 完整的 sanity calibration 执行流程

**使用场景**：验证预训练模型在下游数据集上的学习有效性，而不是简单地过拟合训练数据。

---

### `sanity_calibration_v3_impl.py`
**目的**：增强版本的 sanity calibration 实验，改进的分割完整性追踪。

**主要功能**：
- 与 v2 版本类似的扰动实验（标签打乱、边删除、特征dropout）
- 增强的分割(split)级统计：详细跟踪每个few-shot分割中支撑集(support set)和查询集(query set)
- 更细粒度的指标记录：每个分割的准确率、宏/微F1分数等单独记录

**包含函数**：
- `_run_sanity_calibration_v3_legacy(args, parent_directory, sparse=True)` - 增强版 sanity calibration 执行流程

**使用场景**：进行更严格、更细致的模型校准检验，确保模型学到的是有意义的图特征表示。

---

## 数据流

```
execute.py
  ↓ 调用 run_sanity_calibration / run_sanity_calibration_v3
evaluators/*_impl.py
  ↓ 导入共享工具
utils.usp_shared.*（_usp_load_graph, _usp_device, build_or_load_ego_cache 等）
```

## 配置参数（来自 execute.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sanity_calibration` | False | 是否运行 sanity calibration |
| `--sanity_version` | v2 | sanity 版本：v2 或 v3 |
| `--sanity_epochs` | [1, 10, 50] | 进行评估的预训练轮数 |
| `--label_shuffle_seeds` | [0,1,2] | 标签打乱实验的随机种子 |
| `--overwrite_sanity_table` | False | 是否覆盖已有的 sanity 结果CSV |

## 输出

- **CSV 结果文件**：`result/<set_name>.csv`，包含各扰动条件下的性能指标
- **标准输出**：打印评估进度和关键指标

