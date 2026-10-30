#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT0410
export PYTHONWARNINGS="ignore"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="./result/${TS}"
LOG_FILE="${RUN_DIR}/ab_cora_raw9_${TS}.log"
STAGE1_FEATURE_K="${STAGE1_FEATURE_K:-0}"
mkdir -p "$RUN_DIR"

# 【关键】注释掉这行，禁止输出到终端
# exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run] log file: $LOG_FILE"
echo "[run] result dir: $RUN_DIR"

# 正确：后台运行 + 日志写入你的 LOG_FILE + 终端无输出
nohup python ./src/train/execute.py \
    --dataset Cora \
    --result_subdir "$TS" \
    --pretrain_datasets Citeseer Pubmed \
    --pretrain_method EGRAPHCL \
    --gpu 0 \
    --skip_pretrain 0 \
    --force_rebuild_stage1 1 \
    --force_rebuild_stage2 1 \
    --stage1_num_clusters 8 \
    --stage1_subgraph_hop_n 2 \
    --stage1_encoding_mode raw9 \
    --stage1_feature_k "${STAGE1_FEATURE_K}" \
    --stage1_use_log1p 1 \
    --stage1_remove_selfloops 1 \
    --stage2_top_m_per_cluster 5 \
    --stage2_alpha 1.0 \
    --stage2_beta 1.0 \
    --stage2_gamma 1.0 \
    --stage2_top_ratio 0.3 \
    --pretrain_batch_size 64 \
    --max_subgraph_nodes 64 \
    --context_outputs_root "./result/${TS}/stage_outputs" \
    >> "$LOG_FILE" 2>&1 &