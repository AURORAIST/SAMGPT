#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT0415
export PYTHONWARNINGS="ignore"
BASE_TS="$(date +%Y%m%d_%H%M%S)"
MASK_LIST="${MASK_LIST:-123456789}"
# MASK_LIST="${MASK_LIST:-12567 12578 12579}"
# MASK_LIST="${MASK_LIST:-1235678 1235679  }"
# MASK_LIST="${MASK_LIST:-12 123 124 125 126 127 128 129 1234 12345 123456 123456 1234567 12345678 123456789}"

# 整个脚本用 nohup 丢后台，串行执行所有任务
nohup bash -c '
BASE_TS="$1"
MASK_LIST="$2"

for MASK in ${MASK_LIST}; do
  RUN_ID="${BASE_TS}_raw9m${MASK}"
  RUN_DIR="./result/${RUN_ID}"
  LOG_FILE="${RUN_DIR}/ab_cora_raw9m${MASK}_${BASE_TS}.log"
  mkdir -p "${RUN_DIR}"

  echo "[ablation] start mask=${MASK} run_id=${RUN_ID}"
  echo "[ablation] log file: ${LOG_FILE}"

  # 串行：必须等这个跑完才下一个
  python ./src/train/execute.py \
    --dataset Cora \
    --result_subdir "${RUN_ID}" \
    --pretrain_datasets Photo Computers \
    --pretrain_method EGRAPHCL \
    --gpu 0 \
    --skip_pretrain 0 \
    --force_rebuild_stage1 1 \
    --force_rebuild_stage2 1 \
    --force_rebuild_global_pool 1 \
    --rebuild_run_cache 1 \
    --stage1_num_clusters 8 \
    --stage1_subgraph_hop_n 2 \
    --stage1_encoding_mode raw9 \
    --stage1_feature_k 0 \
    --stage1_feature_mask "${MASK}" \
    --stage1_use_log1p 1 \
    --stage1_remove_selfloops 1 \
    --stage2_top_m_per_cluster 5 \
    --stage2_alpha 1.0 \
    --stage2_beta 1.0 \
    --stage2_gamma 1.0 \
    --stage2_top_ratio 0.3 \
    --pretrain_batch_size 128 \
    --pretrain_max_epochs 100 \
    --pretrain_patience 20 \
    --pretrain_min_delta 1e-4 \
    --egraphcl_negative_pair_mode topk \
    --egraphcl_negative_topk 32 \
    --max_subgraph_nodes 64 \
    --context_outputs_root "./result/${RUN_ID}/stage_outputs" \
    > "${LOG_FILE}" 2>&1

  echo "[ablation] done mask=${MASK}"
  echo
done

echo "[ablation] all runs finished."
' bash "${BASE_TS}" "${MASK_LIST}" > "./run_all_${BASE_TS}.log" 2>&1 &
echo "✅ 已后台稳定运行（nohup 保护，关终端不影响）"
echo "📜 总日志：./run_all_${BASE_TS}.log"