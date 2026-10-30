#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT
export PYTHONWARNINGS="ignore"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="./result/${TS}"
LOG_FILE="${RUN_DIR}/ab_cora_raw9_${TS}.log"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] log file: $LOG_FILE"
echo "[run] result dir: $RUN_DIR"

# Stage outputs will be auto-prepared inside execute.py.
# Tune stage1/stage2 parameters directly via flags below.
python ./src/train/execute.py \
    --dataset Cora \
    --result_subdir "$TS" \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method GRAPHCL \
    --gpu 0 \
    --skip_pretrain 0 \
    --force_rebuild_stage1 1 \
    --force_rebuild_stage2 1 \
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
    --struct_max_subgraph_nodes 64 \
    --context_outputs_root "./result/${TS}/stage_outputs"