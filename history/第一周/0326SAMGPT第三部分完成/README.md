# SAMGPT
We provide the code (in pytorch) and datasets for our paper [**"SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation"**](https://arxiv.org/abs/2502.05424), 
which is accepted by WWW2025.

## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **cache/**: stores intermediate results during pre-training.
- **checkpoints/**: contains checkpoints of the models after pre-training.
- **result/**: stores results when downstream task is done.
- **src/**: implements pre-training and downstream task.

## Package Dependencies

- python 3.8.16
- pytorch 1.10.1
- cuda 11.3
- pyG 2.1.0

## Running experiments
### Data Preparation
To download the datasets and generate fewshot data, run `python ./src/utils/generate_fewshot.py`

### Stage-1: Dataset Definition and Structural Encoding
To run the first step described in `src/context.ipynb` (subgraph construction + structural encoding + KMeans prototypes), use:

```bash
python ./src/context_pipeline/run_stage1.py \
  --datasets Cora Citeseer \
  --output-root ./src/context_pipeline/outputs \
  --max-subgraph-nodes 64 \
  --kmeans-k 8
```

Outputs are saved under `./src/context_pipeline/outputs/{dataset}/`:
- `structural_encodings.csv`: one row per subgraph with 9-D structure features
- `kmeans_labels.npy`: cluster assignment per subgraph
- `kmeans_centers.npy`: prototype centers (in raw feature space)
- `stage1_summary.json`: run metadata and output paths

For Planetoid datasets (`Cora/Citeseer/Pubmed`), Stage-1 first reads local raw files under
`data/{Dataset}/raw/ind.{dataset}.{x,y,tx,ty,allx,ally,graph,test.index}`.
If raw parsing fails, it falls back to the existing PyG loader. This avoids common `.pt`
cache format issues across different PyG versions.

### Node Classification and Graph Classification
To train and evaluate SAMGPT on various target domains, you may need to change the corresponding parameters in *execute.py*.

### Integrated Context Pipeline (Stage-1 + Stage-2 + Training)
`src/execute.py` now always uses the unified context pipeline and automatically prepares artifacts:

- Stage-1 structural encoding (`structural_encodings.csv`, `kmeans_centers.npy`)
- Stage-2 learnability scoring (`learnability_scores.pt`, `representative_pool.pt`)

Run with one command:

```bash
python ./src/execute.py \
  --dataset Cora \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --pretrain_method GRAPHCL \
  --skip_pretrain 0 \
  --context_outputs_root ./src/context_pipeline/outputs
```

Useful control flags:
- `--force_rebuild_stage1 1`: regenerate stage1 outputs
- `--force_rebuild_stage2 1`: regenerate stage2 outputs
- `--stage1_kmeans_k`: stage1 cluster count
- `--stage2_top_m_per_cluster`: stage2 representative samples per cluster
- `--stage2_alpha/--stage2_beta/--stage2_gamma`: stage2 learnability weights

Pre-training and prompt tuning:
```bash
python ./src/execute.py 
  --dataset {Name of the downstream dataset, default: Cora} 
  --pretrain_datasets {List of pretrain datasets, default: ['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']} 
  --downstream_task {Type of downstream task, options: node or graph, default: node}
  --gpu {GPU device number, default: 0} 
  --shot_num {Number of examples for few-shot learning, default: 1} 
  --pretrain_method {Pretraining method, options: GRAPHCL, LP, or splitLP, default: GRAPHCL}  
```

## Mini-batch implementation ##

An example for mini-batch implementation is shown in ./src_batch. 


在 Citeseer 的结构编码里，部分子图出现了 density > 1。
这不是聚类算法问题，而是编码前图里含自环，导致简单图密度定义被破坏。

基线预训练原理（按实际代码顺序）

读取多个预训练数据集，做统一特征处理
在 execute.py 中先调用 process_tu，再做 PCA 压到 unify_dim，缓存 feature 和 adj。

按预训练任务构造训练信号
如果是 GRAPHCL，走 aug.py 的 build_aug，生成两份增强图和对比标签。
如果是 LP 或 splitLP，走 preprompt 里的负采样 tuples（正负对比）。

邻接归一化并加自环
在 execute 里对 adj 做 normalize_adj(adj + I)，这是给 GNN 前向用的标准图卷积预处理。

用 PrePrompt 进行预训练
模型在 preprompt.py：

主干是 GCN/GAT 编码器
两类 prompt：feature prompt 与 structure prompt
loss 根据任务分 GRAPHCL 或 LP 路径
早停并保存 checkpoint
按预训练 loss 做 early stopping，保存最优参数。