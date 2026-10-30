#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT
export PYTHONWARNINGS="ignore"

TS="$(date +%Y%m%d_%H%M%S)"
ROOT_DIR="./result/stability_${TS}"
CONTEXT_ROOT="./result/stage_outputs_fixed/raw9"
mkdir -p "$ROOT_DIR" "$CONTEXT_ROOT"

echo "[stability] root dir: $ROOT_DIR"
echo "[stability] fixed context root: $CONTEXT_ROOT"

for RUN_ID in 1 2 3; do
  RUN_NAME="run${RUN_ID}"
  RUN_DIR="${ROOT_DIR}/${RUN_NAME}"
  LOG_FILE="${RUN_DIR}/ab_cora_raw9_${RUN_NAME}.log"
  mkdir -p "$RUN_DIR"

  echo "[stability] ===== ${RUN_NAME} start ====="
  echo "[stability] log: $LOG_FILE"

  python ./src/train/execute.py \
    --dataset Cora \
    --result_subdir "stability_${TS}/${RUN_NAME}" \
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
    --struct_max_subgraph_nodes 64 \
    --context_outputs_root "$CONTEXT_ROOT" \
    2>&1 | tee "$LOG_FILE"

  echo "[stability] ===== ${RUN_NAME} done ====="
done

echo "[stability] all runs completed"
