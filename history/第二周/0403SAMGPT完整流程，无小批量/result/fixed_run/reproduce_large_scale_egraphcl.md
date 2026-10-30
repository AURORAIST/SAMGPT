# Large-Scale EGRAPHCL Reproduction Notes

This file records the stability and scalability changes for large-sample contrastive pretraining.

## Scope
- Keep global sample pool small (`selected=139` in current setup) to control runtime.
- Improve training stability and scalability without changing dataset split logic.

## Code Changes

### 1) Scalable sample-pair contrastive loss
File: `src/train/preprompt.py`

Implemented a scalable loss path in `PrePrompt._sample_pair_contrastive_loss`:
- Candidate-bank negatives: sample a candidate pool instead of full all-pairs negatives.
- Chunked anchor computation: compute loss in anchor chunks to reduce peak memory.
- Top-k soft-positive masking: keep only strongest `k` soft positives per anchor for masking.
- Negative cap: keep at most `max_negatives` hardest negatives per anchor.

New constructor knobs in `PrePrompt`:
- `egraphcl_temperature`
- `egraphcl_pair_chunk_size`
- `egraphcl_candidate_pool`
- `egraphcl_soft_topk`
- `egraphcl_max_negatives`

### 2) Training stability controls
File: `src/train/execute.py`

Added CLI args:
- `--grad_clip_norm`
- `--lr_scheduler_patience`
- `--lr_scheduler_factor`
- `--lr_scheduler_min_lr`
- `--early_stop_min_delta`

Added EGRAPHCL scalable loss args:
- `--egraphcl_temperature`
- `--egraphcl_pair_chunk_size`
- `--egraphcl_candidate_pool`
- `--egraphcl_soft_topk`
- `--egraphcl_max_negatives`

Training loop updates:
- Non-finite loss guard.
- Gradient clipping (`clip_grad_norm_`).
- `ReduceLROnPlateau` scheduler on train loss.
- Early-stop improvement uses `best - early_stop_min_delta`.

### 3) Run script normalization
File: `run.sh`

- Removed invalid/deprecated args.
- Set method to `EGRAPHCL`.
- Kept small sample-pool setting (`--stage2_top_ratio 0.001`).
- Added stable/scalable defaults.
- Output fixed to `result/fixed_run`.

## Reproduce

Run:

```bash
cd /home/ustc1958/lxy/graph/1/SAMGPT
bash run.sh
```

Main outputs:
- Log: `result/fixed_run/run.log`
- Stage outputs: `result/fixed_run/stage_outputs/`
- Soft-mask figures:
  - `result/fixed_run/egraphcl_soft_mask_by_sample_id.png`
  - `result/fixed_run/egraphcl_soft_mask_by_cluster_id.png`

## Flow Completeness Check (recommended first)

Before long training or increasing sample capacity, run a flow-only validation:

```bash
cd /home/ustc1958/lxy/graph/1/SAMGPT
python ./src/train/execute.py \
  --dataset Cora \
  --result_subdir fixed_run \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --pretrain_method EGRAPHCL \
  --gpu 0 \
  --skip_pretrain 0 \
  --global_subgraph_pool 1 \
  --force_rebuild_global_pool 0 \
  --force_rebuild_stage1 0 \
  --force_rebuild_stage2 0 \
  --stage1_num_clusters 8 \
  --stage1_subgraph_hop_n 2 \
  --stage1_encoding_mode raw9 \
  --stage1_use_log1p 1 \
  --stage1_remove_selfloops 1 \
  --stage2_top_m_per_cluster 8 \
  --stage2_alpha 1.0 \
  --stage2_beta 1.0 \
  --stage2_gamma 1.0 \
  --stage2_top_ratio 0.001 \
  --egraphcl_tau 0.85 \
  --egraphcl_temperature 0.2 \
  --egraphcl_pair_chunk_size 256 \
  --egraphcl_candidate_pool 1024 \
  --egraphcl_soft_topk 10 \
  --egraphcl_max_negatives 512 \
  --max_subgraph_nodes 64 \
  --flow_check_only 1 \
  --flow_report_name flow_check_report.json \
  --context_outputs_root ./result/fixed_run/stage_outputs
```

Flow check output:
- `result/fixed_run/flow_check_report.json`

The report includes pass/fail checks for:
- Stage artifacts presence.
- Augmentation tensor alignment.
- Dataset and cluster coverage in selected samples.
- EGRAPHCL pair-mask shape and usable negative pairs.

## Recommended Knob Ranges (large-scale)
- `egraphcl_pair_chunk_size`: 128 ~ 512
- `egraphcl_candidate_pool`: 512 ~ 4096
- `egraphcl_soft_topk`: 5 ~ 20
- `egraphcl_max_negatives`: 256 ~ 2048
- `lr`: 1e-4 ~ 5e-4 with scheduler + grad clip

## Notes
- Candidate-pool approximation changes the exact loss value scale. Compare trends and downstream metrics, not absolute loss magnitudes only.
- If runtime is still high, reduce `egraphcl_candidate_pool` first.
- If convergence is too slow, increase `egraphcl_candidate_pool` before raising `lr`.
