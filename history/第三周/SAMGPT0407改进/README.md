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
To run the first step described in `src/context.ipynb` (2-hop subgraph construction + raw9 structural encoding + GMM clustering), use:

```bash
python ./src/train/run_stage_pipeline.py \
  --stage 1 \
  --datasets Cora Citeseer \
  --output-root ./result/stage_outputs/stage1 \
  --subgraph-hop-n 2 \
  --encoding-mode raw9 \
  --max-subgraph-nodes 64 \
  --num-clusters 8
```

Outputs are saved under `./result/stage_outputs/stage1/{dataset}/`:
- `structural_encodings.csv`: one row per subgraph with 9-D structure features
- `kmeans_labels.npy`: cluster assignment per subgraph
- `kmeans_centers.npy`: prototype centers (in raw feature space)
- `stage1_summary.json`: run metadata and output paths

For Planetoid datasets (`Cora/Citeseer/Pubmed`), Stage-1 first reads local raw files under
`data/{Dataset}/raw/ind.{dataset}.{x,y,tx,ty,allx,ally,graph,test.index}`.
If raw parsing fails, it falls back to the existing PyG loader. This avoids common `.pt`
cache format issues across different PyG versions.

### Node Classification and Graph Classification
To train and evaluate SAMGPT on various target domains, you may need to change the corresponding parameters in *src/train/execute.py*.

### Integrated Context Pipeline (Stage-1 + Stage-2 + Training)
`src/train/execute.py` now always uses the unified context pipeline and automatically prepares artifacts:

- Stage-1 structural encoding (`structural_encodings.csv`, `cluster_centers.npy`)
- Stage-2 two-step filtering (`learnability_scores.pt`, `representative_pool.pt`)
- Stage-2 debug and visualization (`learnability_summary.json`, `stage2_selection_visualization.png`)
- Stage-2 decoded pool matrices (`decoded_subgraphs/sample_*.npz`, `dataset_prompt.npy`)

Run with one command:

```bash
python ./src/train/execute.py \
  --dataset Cora \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --pretrain_method GRAPHCL \
  --skip_pretrain 0 \
  --context_outputs_root ./result/stage_outputs
```

Useful control flags:
- `--force_rebuild_stage1 1`: regenerate stage1 outputs
- `--force_rebuild_stage2 1`: regenerate stage2 outputs
- `--stage2_top_m_per_cluster`: stage2 representative samples per cluster
- `--stage2_top_ratio`: stage2 second-selection ratio on remaining samples
- `--stage2_alpha/--stage2_beta/--stage2_gamma`: stage2 learnability weights
- `--egraphcl_batch_size`: mini-batch size for sample-level negative contrast (default 128)

Pre-training and prompt tuning:
```bash
python ./src/train/execute.py 
  --dataset {Name of the downstream dataset, default: Cora} 
  --pretrain_datasets {List of pretrain datasets, default: ['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']} 
  --downstream_task {Type of downstream task, options: node or graph, default: node}
  --gpu {GPU device number, default: 0} 
  --shot_num {Number of examples for few-shot learning, default: 1} 
  --pretrain_method {Pretraining method, options: GRAPHCL, EGRAPHCL, LP, or splitLP, default: GRAPHCL}  
```

## Mini-batch implementation ##

An example for mini-batch implementation is shown in ./src_batch. 


在 Citeseer 的结构编码里，部分子图出现了 density > 1。
这不是聚类算法问题，而是编码前图里含自环，导致简单图密度定义被破坏。

基线预训练原理（按实际代码顺序）

读取多个预训练数据集，做统一特征处理
在 src/train/execute.py 中先调用 process_tu，再做 PCA 压到 unify_dim，缓存 feature 和 adj。

按预训练任务构造训练信号
如果是 GRAPHCL，走 aug.py 的 build_aug，生成两份增强图和对比标签。
如果是 LP 或 splitLP，走 preprompt 里的负采样 tuples（正负对比）。

邻接归一化并加自环
在 src/train/execute.py 里对 adj 做 normalize_adj(adj + I)，这是给 GNN 前向用的标准图卷积预处理。

用 PrePrompt 进行预训练
模型在 preprompt.py：

主干是 GCN/GAT 编码器
两类 prompt：feature prompt 与 structure prompt
loss 根据任务分 GRAPHCL 或 LP 路径
早停并保存 checkpoint
按预训练 loss 做 early stopping，保存最优参数。

/home/ustc1958/lxy/graph/1/SAMGPT这是我目前的项目路径，请帮我重写梳理代码。run.sh为启动脚本。首先每个数据集的图构建2-hop获得若干子图，每个子图应该记录属于哪个图，然后将每个子图利用raw9方法进行编码，将所有子图的编码结果聚集成一个大样本集合，然后对这个集合进行聚类。同时设定聚类数，这个被聚类的集合为初始样本池，同时，在调试输出中打印每个簇的样本个数，并可视化同时标出样本中心。这是第一阶段。在第二阶段，我们利用这个初始样本池进行筛选，首先我们选取每个簇的前m个样本，作为一阶段筛选，其次，我们分别计算Coverage，Novelty，和Redundancy分数并聚合，选出剩余样本池前p%的样本，作为第二次筛选，两次筛选出的样本汇总为一个代表池。接着，将这个样本池的样本进行解码，返回得到初始的子图表示，并计算特征矩阵和邻接矩阵。注意这里的调试信息比较多，要输出一二次筛选个数，总个数，以及每次筛选每个簇的占比。同时输出筛选后的可视化，与之前聚类的可视化要求一致，便于对比聚类的效果。最后，根据邻接矩阵和特征矩阵，依次进行pca降维，去自环，归一化等操作，特征矩阵加上第一个prompt，这个prompt与数据集有关，相同数据集共享一个prompt，以便对其多领域数据集。然后共同构建一个正样本增强视图。负样本应该选取小批量的方法，设立超参默认128，将其中每一个样本与批次内其他样本进行负对比学习，每轮批次随机构建，并在每个小批次内计算样本间相似图，如果样本相似度高于阈值则屏蔽，否则为负样本。这里也给出统计告诉我屏蔽是否生效。最后，给编码器也加上prompt进行调整，分批次进行训练。1/SAMGPT/unsupervised_TU记录了graphcl再多图训练时的方法，你可以参考。