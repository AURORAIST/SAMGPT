#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPTstreaming
export PYTHONWARNINGS="ignore"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="./result/${TS}"
STAGE_CACHE_DIR="/home/ustc1958/lxy/graph/1/SAMGPTstreaming/result/stage_outputs_shared"
LOG_FILE="${RUN_DIR}/ab_cora_raw9_${TS}.log"
mkdir -p "$RUN_DIR"
mkdir -p "$STAGE_CACHE_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] log file: $LOG_FILE"
echo "[run] result dir: $RUN_DIR"
echo "[run] stage cache dir: $STAGE_CACHE_DIR"
echo "[run] stage rebuild disabled; reusing existing stage outputs when present"

# Stage outputs will be auto-prepared inside execute.py.
# Tune stage1/stage2 parameters directly via flags below.
python ./src/train/execute.py \
    --dataset Cora \
    --result_subdir "$TS" \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method GRAPHCL \
    --gpu 0 \
    --skip_pretrain 0 \
    --force_rebuild_stage1 0 \
    --force_rebuild_stage2 0 \
    --stage1_num_clusters 8 \
    --stage1_subgraph_hop_n 2 \
    --stage1_encoding_mode raw9 \
    --stage1_use_log1p 1 \
    --stage1_remove_selfloops 1 \
    --stage2_top_m_per_cluster 5 \
    --stage2_alpha 1.0 \
    --stage2_beta 1.0 \
    --stage2_gamma 1.0 \
    --stage2_top_ratio 0.3 \
    --stage2_max_subgraph_nodes 64 \
    --graphcl_online_subgraph 1 \
    --graphcl_online_max_subgraphs_per_dataset 800 \
    --graphcl_online_samples_per_dataset_per_epoch 0 \
    --graphcl_use_amp 1 \
    --streaming_grad_accum_steps 16 \
    --streaming_micro_batch_size 16 \
    --streaming_grad_clip_norm 1.0 \
    --graphcl_split_max_nodes 0 \
    --graphcl_split_shuffle_each_epoch 1 \
    --struct_max_subgraph_nodes 64 \
    --context_outputs_root "$STAGE_CACHE_DIR"