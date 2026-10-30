#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT
export PYTHONWARNINGS="ignore"

RUN_SUBDIR="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="./result/${RUN_SUBDIR}"
LOG_FILE="${RUN_DIR}/run.log"
NOHUP_LOG_FILE="${RUN_DIR}/nohup.log"
mkdir -p "$RUN_DIR"
exec > >(tee "$LOG_FILE") 2>&1
echo "[run] log file: $LOG_FILE"
echo "[run] result dir: $RUN_DIR"
echo "[run] nohup log file: $NOHUP_LOG_FILE"

nohup python ./src/train/execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method EGRAPHCL \
    --downstream_task node \
    --gpu 0 \
    --skip_pretrain 0 \
    --graphId 1 \
    --feature_align_method pca \
    --ablation_pre st \
    --lr 0.0003 \
    --egraphcl_tau 0.85 \
    --egraphcl_adaptive_tau 1 \
    --egraphcl_target_mask_ratio 0.35 \
    --egraphcl_tau_min 0.82 \
    --egraphcl_tau_max 0.99 \
    --egraphcl_temperature 0.2 \
    --egraphcl_batch_size 128 \
    --egraphcl_batch_max_domains 3 \
    --egraphcl_batch_max_domain_ratio 0.5 \
    --egraphcl_min_negatives_per_anchor 12 \
    --egraphcl_max_mask_ratio 0.55 \
    --egraphcl_keep_data_on_cpu 1 \
    --egraphcl_epoch_chunk_graphs 128 \
    --subgraph_hop_n 2 \
    --max_subgraph_nodes 64 \
    --encoder_filter_enable 1 \
    --encoder_filter_warmup_epochs 200 \
    --encoder_filter_method gmm \
    --encoder_filter_clusters 8 \
    --encoder_filter_top_m_per_cluster 8 \
    --encoder_filter_top_ratio 0.3 \
    --encoder_filter_score_alpha 1.0 \
    --encoder_filter_score_beta 1.0 \
    --encoder_filter_score_gamma 1.0 \
    --encoder_filter_visualize 1 \
    --encoder_filter_visualize_max_points 50000 \
    --debug_trace_enable 1 \
    --encoder_filter_force_rebuild 1 \
    --result_subdir "$RUN_SUBDIR" \
    --context_outputs_root "./result/${RUN_SUBDIR}/encoder_outputs" \
    > "$NOHUP_LOG_FILE" 2>&1 &

echo "[run] started in background, pid=$!"