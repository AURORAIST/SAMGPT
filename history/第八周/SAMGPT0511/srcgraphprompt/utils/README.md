# utils 文件夹

工具函数（utilities）模块。包含数据加载、图处理、特征增强、共享助手等基础功能，是整个项目的支撑设施。

## 文件清单

### `__init__.py`
包初始化文件，导出该模块的公开接口。

---

### `dataset.py`
**目的**：数据集加载和初始化。

**主要功能**：
- `load_dataset(name, path=...)` - 加载指定名称的图数据集
- 支持的数据集：
  - 社交网络：Cora, Citeseer, Pubmed
  - 商品图：Computers, Photo
  - 音乐网络：LastFMAsia
  - 页面链接：FacebookPagePage
  - 其他 TU 数据集（通过 PyTorch Geometric 加载）
- 数据预处理：顶点特征和邻接矩阵初始化
- 缓存管理：避免重复下载和处理

**返回类型**：PyTorch Geometric `Data` 对象或类似的图表示

**使用场景**：
- 预训练时加载多个源数据集
- 下游任务时加载目标数据集

---

### `process.py`
**目的**：图数据的预处理和格式转换。

**主要功能**：

#### 特征处理
- `process_tu(data, input_dim)` - 处理 TU 数据集格式，提取特征和邻接
- 特征归一化：z-score 或其他标准化
- 特征拼接：多种特征源的融合

#### 邻接处理
- `normalize_adj(adj)` - 对称归一化邻接矩阵（D^-1/2 * A * D^-1/2）
- `combine_dataset_list_sp(adjs)` - 多个邻接矩阵的块对角合并（用于 LP 预训练）
- 自环处理：添加或移除自环

#### 矩阵格式转换
- `sparse_mx_to_torch_sparse_tensor(adj)` - SciPy 稀疏矩阵 → PyTorch 稀疏张量
- 支持 COO、CSR、CSC 等稀疏格式的互转

#### 子图提取
- `build_subgraph(adj_dense, target_indices, sparse, subgraph_hop, max_neighbors)` - 构建目标节点的 k-hop 子图邻接和节点索引

**使用场景**：
- 特征 PCA 压缩前的数据规范化
- 邻接矩阵的规范化和格式转换
- 下游 few-shot 任务中的子图提取

---

### `aug.py`
**目的**：图数据增强（graph augmentation）。

**主要功能**：

#### 增强策略
- `aug_random_mask(input_feature, drop_percent)` - 随机特征遮蔽（feature masking）
  - 将部分节点的特征置零，模拟缺失特征
  
- `aug_random_drop(input_adj, drop_percent)` - 随机边删除（edge dropping）
  - 随机选择边并删除，改变图拓扑
  
- `aug_subgraph(input_adj, drop_percent)` - 随机子图抽取（subgraph sampling）
  - 从图中采样连通子图
  
- `build_aug(adj, feature, sparse, drop_percent, aug_type)` - 批量增强生成
  - 生成两份或多份增强视图
  - 支持边增强模式（`aug_type='edge'`）和特征增强模式（`aug_type='mask'`）
  - 返回增强特征、增强邻接和对应的对比标签

**使用场景**：
- GraphCL 对比预训练：生成两份增强视图
- DGI 预训练：生成打乱的对比视图
- 鲁棒性测试：在下游数据上应用增强验证模型鲁棒性

---

### `usp_shared.py`
**目的**：USP 相关的共享工具函数集合，供预训练和评测代码复用。

**主要功能**：

#### 设备和计算
- `_usp_device(args)` - 解析设备参数（cuda/cpu/auto）

#### 图加载和缓存
- `_usp_load_graph(dataset_name, cache_dir, unify_dim, args)` - 加载图特征和邻接，进行 PCA 压缩到统一维度
- `build_or_load_ego_cache(args, adj_sp, cache_dir, dataset_name, subgraph_hop, max_neighbors, sparse)` - 构造或加载 k-hop ego-subgraph 的 CSR 掩码
  - 块级缓存(block caching)：分块存储大型图的子图信息，节省内存
  - 掩码复用：避免重复计算相同配置的子图

#### Few-shot 数据读取
- `_usp_read_fewshot(parent_directory, dataset_name, shot_num, split_id, sparse)` - 读取预定义的 few-shot 分割
  - 支持节点和图两种任务类型
  - 返回支撑集、查询集索引

#### 特征增强
- `_usp_feature_dropout(features, drop_percent)` - 特征级别的 dropout
- `_usp_edge_dropout(adj, drop_percent)` - 边级别的 dropout

#### 图操作和指标
- `_permute_target_edges(adj_sp, permute_percent)` - 随机排列目标图的边
- `_identity_like_adj(adj)` - 创建与邻接矩阵兼容的单位矩阵
- `_validate_neg_exclude_scope(neg_exclude_scope)` - 验证负采样排除范围参数

#### 结果输出
- `_write_sanity_summary(output_file, epoch, method_name, metric_dict)` - 向 CSV 写入 sanity 指标
- `_write_rows_csv(csv_path, rows)` - 通用 CSV 行写入函数
- `_build_result_dict(epoch, split, shot, acc, macro_f1, micro_f1, ...)` - 构建结果字典，用于 CSV 记录

#### 多数据源融合
- 可能包含多源负采样的辅助函数：域平衡抽样、域判别等

**包含函数总数**：约 20+ 个实用函数

**使用场景**：
- 预训练（trainers/usp_minimal_impl.py）中复用图加载、子图缓存、特征增强
- 下游评测（evaluators/*_impl.py）中复用 few-shot 读取、特征增强、结果输出
- 实验运行（experiments/*_impl.py）中复用指标计算和日志写入

---

### `execute_helpers.py`
**目的**：主程序（execute.py）的助手函数，降低 execute.py 的代码复杂度。

**主要功能**：

#### 设备和文件安全
- `resolve_device(device_str, gpu_id)` - 解析设备参数并返回 PyTorch device 对象
- `torch_load_trusted(file_path)` - 安全的 torch.load（可防止某些恶意序列化）
- `load_state_dict_trusted(file_path)` - 安全加载模型权重

#### 字符串解析
- `str2bool(v)` - 将字符串"True"/"False"转换为布尔值

#### 下游评测
- `prepare_downstream_eval_context(args, parent_directory, device, unify_dim, sparse)` - 预准备下游任务的上下文，包括模型构建、优化器初始化等
- `run_downstream_validation(args, model, context, device, parent_directory, pretrain_dataset_str, shot_num, ...)` - 在预训练中周期性地运行下游验证

**使用场景**：
- execute.py 中参数解析和安全加载
- 下游评测中的模型和数据准备
- 混合精度训练中的设备检测

---

### `generate_fewshot.py`
**目的**：生成 few-shot 评测的分割（splits）文件。

**主要功能**：
- 为目标数据集生成多个固定的 few-shot 分割
- 每个分割包含：支撑集（support set）和查询集（query set）
- 支持类型：
  - 节点分类(node-level)：从目标节点中采样部分作为支撑集
  - 图分类(graph-level)：从子图中采样相似图作为支撑集
- 固定种子以确保可复现性
- 输出格式：PyTorch 张量文件（.pt）

**包含函数**：
- `generate_fewshot_splits(dataset, num_splits, shots, ...)` 或类似 - 生成并保存分割文件

**使用场景**：
- 一次性生成 few-shot 分割（在实验前运行）
- 与其他代码共享相同的分割，确保评估的一致性

**注意**：该脚本通常作为独立脚本运行一次，然后其他代码读取其输出

---

## 模块依赖关系

```
utils/
├─ dataset.py ← 被 execute.py 和 *_impl.py 调用
├─ process.py ← 被 execute.py、**_impl.py 和 aug.py 调用
├─ aug.py ← 被 execute.py 和 *_impl.py 调用（用于 GraphCL 增强）
├─ usp_shared.py ← 被 trainers/usp_minimal_impl.py 和 evaluators/*_impl.py 调用
├─ execute_helpers.py ← 被 execute.py 调用
└─ generate_fewshot.py ← 独立脚本，生成 few-shot 分割文件
```

## 配置和初始化

基础配置通常来自 execute.py 的参数：
- `--unify_dim` - PCA 压缩的目标维度（默认 50）
- `--drop_percent` - 增强时的删除比例（默认 0.1）
- `--aug_type` - 增强类型："edge" 或 "mask"（默认 "edge"）
- `--subgraph_hop` - 子图提取的跳数（默认 2）
- `--max_neighbors` - 子图中每节点的邻接上限（默认 64）

## 使用示例

```python
# 数据加载
from utils.dataset import load_dataset
data = load_dataset('Cora', path='./data')

# 特征处理和 PCA
from utils.process import process_tu
from preprompt import pca_compression
features, adj = process_tu(data, data.x.shape[1])
features = pca_compression(features, k=50)

# 邻接归一化
from utils.process import normalize_adj
adj = normalize_adj(adj + sp.eye(adj.shape[0]))

# 对比增强
from utils.aug import build_aug
aug_features, aug_adjs, labels = build_aug(adj, features, sparse=True, drop_percent=0.1)

# 子图提取
from utils.process import build_subgraph
subgraph = build_subgraph(adj.todense().A, indices, sparse=False, subgraph_hop=2)
```

