# USP-SAM Experiment Protocol

This file is a result-free protocol. Every metric cell is intentionally `TBD`
until produced by controlled runs.

## 1. Full Experimental Setup

| Item | Setting |
| --- | --- |
| Method name | USP-SAM: Unified Subgraph Prompting for SAMGPT |
| Core hypothesis | Align pretraining and downstream adaptation through node-subgraph and subgraph-subgraph similarity |
| Source domains | Citation: Cora, Citeseer, Pubmed; Co-purchase: Photo, Computers; Social: FacebookPagePage, LastFMAsia; protein/graph-level datasets if available |
| Target protocol | Leave-one-domain/dataset-out target adaptation |
| Feature unification | PCA or linear projection to `unify_dim` |
| Encoder | Shared GCN/GAT/GraphSAGE encoder; SAMGPT GCN kept as default |
| Structure token | Use SAMGPT structure prompt/token by default; ablate with `without structure token` |
| Ego-subgraph | k-hop ego graph, default `k=2`; ablate `k in {1,2,3}` |
| Readout | Mean, attention, prompt-weighted |
| Pretraining loss | `L_pre = L_ns + lambda_1 L_ss + lambda_2 L_align + lambda_3 L_reg` |
| Downstream classifier | Class prototype subgraph; mean and attention prototype variants |
| Query modes | Node-query, subgraph-query, hybrid-query |
| Shots | 1-shot, 3-shot, 5-shot, 10-shot |
| Seeds/splits | Use at least 10 seeds; keep current 100 few-shot splits where available |
| Main metrics | Accuracy, Macro-F1, Micro-F1 |
| Transfer metrics | Source-to-target transfer gap, average target rank, mean +/- std |
| Representation metrics | Prototype compactness/separation, domain discrepancy, query-prototype margin |

## 2. Baseline Comparison Table

| Method | Pretraining objective | Downstream adaptation | Structure alignment | Acc | Macro-F1 | Micro-F1 | Mean rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GCN supervised fine-tuning | None | MLP/classifier fine-tune | No | TBD | TBD | TBD | TBD |
| GAT supervised fine-tuning | None | MLP/classifier fine-tune | No | TBD | TBD | TBD | TBD |
| GraphSAGE supervised fine-tuning | None | MLP/classifier fine-tune | No | TBD | TBD | TBD | TBD |
| DGI | Node-global discrimination | Linear/prototype eval | No | TBD | TBD | TBD | TBD |
| GraphCL | View-level contrast | Linear/prototype eval | No | TBD | TBD | TBD | TBD |
| GCA | Adaptive graph contrast | Linear/prototype eval | Optional | TBD | TBD | TBD | TBD |
| GraphPrompt | Prompt-based adaptation | GraphPrompt classifier | No | TBD | TBD | TBD | TBD |
| GraphPrompt+ | Enhanced prompt adaptation | GraphPrompt+ classifier | No | TBD | TBD | TBD | TBD |
| SAMGPT | GraphCL/LP + structure prompt | Original SAMGPT prompt adaptation | Yes | TBD | TBD | TBD | TBD |
| SAMGPT + original downstream prompt | Original SAMGPT pretrain | Original prototype-like node prompt | Yes | TBD | TBD | TBD | TBD |
| USP-SAM | Node-subgraph + subgraph-subgraph similarity | Class prototype subgraph | Yes | TBD | TBD | TBD | TBD |

## 3. Ablation Table

| Variant | Purpose | Acc | Macro-F1 | Micro-F1 | Margin | Compactness | Separation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USP-SAM full | Reference | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o subgraph similarity objective | Replace with DGI-style node-global discrimination | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o prompt-weighted readout | Use mean readout | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o class prototype subgraph | Use MLP classifier | TBD | TBD | TBD | TBD | TBD | TBD |
| 1-hop ego-subgraph | Test under-local context | TBD | TBD | TBD | TBD | TBD | TBD |
| 2-hop ego-subgraph | Default balance point | TBD | TBD | TBD | TBD | TBD | TBD |
| 3-hop ego-subgraph | Test over-smoothing/noise | TBD | TBD | TBD | TBD | TBD | TBD |
| node-query | Query is `h_v` | TBD | TBD | TBD | TBD | TBD | TBD |
| subgraph-query | Query is `g_v` | TBD | TBD | TBD | TBD | TBD | TBD |
| hybrid-query | Query is `eta h_v + (1-eta) g_v` | TBD | TBD | TBD | TBD | TBD | TBD |
| without structure token | Remove SAMGPT structure token | TBD | TBD | TBD | TBD | TBD | TBD |
| random negatives | Test shortcut risk | TBD | TBD | TBD | TBD | TBD | TBD |
| domain-balanced negatives | Control domain shortcut | TBD | TBD | TBD | TBD | TBD | TBD |
| mean prototype | Simple class prototype | TBD | TBD | TBD | TBD | TBD | TBD |
| attention prototype | Learned support weighting | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. Few-Shot Result Table

| Target domain | Method | 1-shot Acc/F1 | 3-shot Acc/F1 | 5-shot Acc/F1 | 10-shot Acc/F1 | Stability |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| Cora | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| Citeseer | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| Citeseer | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| Pubmed | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| Pubmed | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| Photo | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| Photo | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| Computers | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| Computers | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| FacebookPagePage | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| FacebookPagePage | USP-SAM | TBD | TBD | TBD | TBD | TBD |
| LastFMAsia | SAMGPT | TBD | TBD | TBD | TBD | TBD |
| LastFMAsia | USP-SAM | TBD | TBD | TBD | TBD | TBD |

## 5. Cross-Domain Transfer Table

| Source domains | Target domain | Structural gap type | SAMGPT Acc/F1 | USP-SAM Acc/F1 | Transfer gap | Rank delta |
| --- | --- | --- | --- | --- | --- | --- |
| All except Cora | Cora | Citation target | TBD | TBD | TBD | TBD |
| All except Citeseer | Citeseer | Citation target | TBD | TBD | TBD | TBD |
| All except Pubmed | Pubmed | Citation target | TBD | TBD | TBD | TBD |
| Citation sources | Photo | Citation to co-purchase | TBD | TBD | TBD | TBD |
| Citation sources | Computers | Citation to co-purchase | TBD | TBD | TBD | TBD |
| Citation + co-purchase | FacebookPagePage | To social | TBD | TBD | TBD | TBD |
| Citation + co-purchase | LastFMAsia | To social | TBD | TBD | TBD | TBD |
| Node datasets | Graph-level dataset | Node-to-graph extension | TBD | TBD | TBD | TBD |

## 6. Visualization Analysis Design

| Figure | Representation | Comparison | Expected evidence if hypothesis holds |
| --- | --- | --- | --- |
| t-SNE/UMAP | Node and ego-subgraph embeddings | SAMGPT vs USP-SAM | Domains mix better while labels form tighter local clusters |
| Prototype distance heatmap | Class prototypes | Mean vs attention prototype; SAMGPT vs USP-SAM | Larger inter-class distance and less prototype collapse |
| Similarity distribution | Pretrain positive/negative similarity and downstream query-prototype similarity | DGI-style vs USP-SAM | Pretrain and downstream score distributions become more consistent |
| Ego-subgraph case study | Support prototypes and high-similarity query subgraphs | Correct vs confused classes | Prototype subgraphs capture stable local structural roles |
| Cross-domain alignment plot | Source/target subgraphs matched by structural role | With vs without structure token | Similar roles across domains become closer without erasing class signal |

## 7. Failure Analysis Checklist

| Symptom | Diagnostic | Likely cause | First fix |
| --- | --- | --- | --- |
| USP-SAM beats weak baselines but not SAMGPT | Pairwise target-domain comparison | Objective not helping transfer | Tune `lambda_ss`, temperature, negative sampling |
| Good pretrain loss but poor few-shot accuracy | Query-prototype margin near zero | Task form still mismatched or prototypes unstable | Use attention prototype, prototype regularization |
| Large gain only within one source family | Domain classifier accuracy high | Domain shortcut in negatives | Use domain-balanced negatives |
| 3-hop collapses performance | Ego-subgraph size statistics, degree buckets | Over-smoothing/noisy neighborhoods | Neighbor cap or prompt-weighted readout |
| 1-shot variance too high | Split-wise std and prototype norm | Prototype instability | Attention prototype, support augmentation |
| Structure token hurts | Domain discrepancy vs class separation | Alignment erases class-specific signal | Lower `lambda_align` or token dropout |

## 8. Results That Can Support The Paper Claim

| Claim | Required evidence |
| --- | --- |
| Unified subgraph similarity improves cross-domain transfer | USP-SAM outperforms SAMGPT on most target domains and shots with mean +/- std |
| Node-subgraph/subgraph-subgraph is better than DGI-style discrimination | `w/o subgraph similarity objective` drops significantly under identical encoder and prompt settings |
| Prompt-weighted readout matters | Prompt-weighted readout beats mean readout, especially for high-degree or noisy target graphs |
| Class prototype subgraph improves adaptation | Prototype subgraph classifier beats MLP classifier under 1/3/5/10-shot settings |
| 2-hop is a practical balance | 2-hop has best or most stable average rank across datasets |
| Hybrid-query is more stable | Hybrid-query has better average rank and lower std than node-query/subgraph-query |
| Structure token remains useful | With-structure-token USP-SAM beats without-token USP-SAM and keeps domain discrepancy controlled |
| Task-form consistency is real | Similarity distribution and margin analyses show better alignment between pretrain and downstream scores |

## 9. Results That Are Not Enough

| Observation | Why insufficient |
| --- | --- |
| USP-SAM beats only GCN/DGI but not SAMGPT | Does not validate the SAMGPT paradigm modification |
| A single target dataset improves | Could be dataset-specific tuning |
| Accuracy improves but Macro-F1 drops | May favor majority classes rather than better transfer |
| Better t-SNE only | Visualization without metric improvement is not enough |
| Prompt readout improves only with more parameters | Need parameter-matched attention/MLP readout control |
| 2-hop wins only on citation graphs | Does not establish cross-structure robustness |
| Structure token helps only source-domain validation | Does not prove target-domain transfer benefit |

## 10. Priority Modules To Modify Next

| Priority | Module | Concrete action |
| --- | --- | --- |
| P0 | Runner integration | Add `--pretrain_method USP` and call `USPPretrainingHead` in the existing training loop |
| P0 | Downstream classifier | Replace node-only prototype with `ClassPrototypeSubgraphClassifier` |
| P1 | Negative sampling | Implement random and domain-balanced negative queues |
| P1 | Neighbor control | Add neighbor sampling cap for 2-hop/3-hop ego-subgraphs |
| P1 | Metrics logger | Log margin, compactness, separation, domain discrepancy |
| P2 | Visualization scripts | Export embeddings/prototypes/similarity scores for UMAP and heatmaps |
| P2 | Graph-level extension | Treat whole graph as a special subgraph and reuse the same prototype template |
