#!/usr/bin/env bash
set -euo pipefail

# ========= 可配参数 =========
REPO_ROOT="${REPO_ROOT:-/home/ustc1958/lxy/graph/1/SAMGPT0415}"
BASE_TS="${BASE_TS:-$(date +%Y%m%d_%H%M%S)}"

MASK_LIST="${MASK_LIST:-123456789}"

# 可选：指定 conda 环境；留空则直接用当前 python
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED="1"

cd "${REPO_ROOT}"

nohup bash -c '
set -euo pipefail

BASE_TS="$1"
MASK_LIST="$2"
CONDA_ENV="$3"
PYTHON_BIN="$4"

MASTER_LOG="./run_all_${BASE_TS}.log"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

echo "[ablation] start base_ts=${BASE_TS}" | tee -a "${MASTER_LOG}"
echo "[ablation] masks=${MASK_LIST}" | tee -a "${MASTER_LOG}"
echo "[ablation] conda_env=${CONDA_ENV:-<none>}" | tee -a "${MASTER_LOG}"

for MASK in ${MASK_LIST}; do
    RUN_ID="${BASE_TS}_raw9m${MASK}"
    RUN_DIR="./result/${RUN_ID}"
    LOG_FILE="${RUN_DIR}/ab_cora_raw9m${MASK}_${BASE_TS}.log"
    mkdir -p "${RUN_DIR}"

    echo "[ablation] start mask=${MASK} run_id=${RUN_ID}" | tee -a "${MASTER_LOG}"
    echo "[ablation] log file: ${LOG_FILE}" | tee -a "${MASTER_LOG}"

    run_python ./src/train/execute.py \
      --dataset Cora \
      --result_subdir "${RUN_ID}" \
      --pretrain_datasets Photo Computers \
      --pretrain_method EGRAPHCL \
      --gpu 0 \
      --skip_pretrain 0 \
      --ablation_pre ft \
      --ablation_down fo \
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
      --egraphcl_encoder_mode batched \
      --prompt_routing_mode dataset \
      --prototype_soft_prompt 1 \
      --downstream_prompt_bank_mode auto \
      --max_subgraph_nodes 64 \
      --context_outputs_root "./result/${RUN_ID}/stage_outputs" \
      > "${LOG_FILE}" 2>&1

    echo "[ablation] done mask=${MASK}" | tee -a "${MASTER_LOG}"
    echo | tee -a "${MASTER_LOG}"
done

echo "[ablation] all runs finished." | tee -a "${MASTER_LOG}"
' bash "${BASE_TS}" "${MASK_LIST}" "${CONDA_ENV}" "${PYTHON_BIN}" \
> "./launcher_${BASE_TS}.log" 2>&1 &

echo "✅ 已后台启动（nohup），关终端不影响"
echo "📜 总日志: ./run_all_${BASE_TS}.log"
echo "📜 启动日志: ./launcher_${BASE_TS}.log"