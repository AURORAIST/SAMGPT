#!/usr/bin/env bash
set -euo pipefail

# ========= 可配参数 =========
REPO_ROOT="${REPO_ROOT:-/home/ustc1958/lxy/graph/1/SAMGPT0422}"
BASE_TS="${BASE_TS:-$(date +%Y%m%d_%H%M%S)}"

MASK_LIST="${MASK_LIST:-123456789}"
ROUTE_LIST="${ROUTE_LIST:-prototype}"
TARGET_DATASET="${TARGET_DATASET:-Cora}"
PRETRAIN_DATASETS="${PRETRAIN_DATASETS:-Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia}"

# 快速验证可通过环境变量覆盖；正式跑大实验时可把 PRETRAIN_MAX_EPOCHS 调回 100
PRETRAIN_MAX_EPOCHS="${PRETRAIN_MAX_EPOCHS:-100}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-50}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
PROTO_ASSIGN_TAU="${PROTO_ASSIGN_TAU:-0.20}"
PROTO_CONSISTENCY_W="${PROTO_CONSISTENCY_W:-0.05}"
PROTO_BALANCE_W="${PROTO_BALANCE_W:-0.01}"
BUILD_TARGET_STAGE1="${BUILD_TARGET_STAGE1:-0}"
TARGET_STAGE1_DATASET="${TARGET_STAGE1_DATASET:-Cora}"
TARGET_STAGE1_FORCE="${TARGET_STAGE1_FORCE:-0}"

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
ROUTE_LIST="$5"
PRETRAIN_MAX_EPOCHS="$6"
PRETRAIN_PATIENCE="$7"
PRETRAIN_BATCH_SIZE="$8"
PROTO_ASSIGN_TAU="$9"
PROTO_CONSISTENCY_W="${10}"
PROTO_BALANCE_W="${11}"
BUILD_TARGET_STAGE1="${12}"
TARGET_STAGE1_DATASET="${13}"
TARGET_STAGE1_FORCE="${14}"
TARGET_DATASET="${15}"
PRETRAIN_DATASETS="${16}"

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
echo "[ablation] routes=${ROUTE_LIST}" | tee -a "${MASTER_LOG}"
echo "[ablation] pretrain_max_epochs=${PRETRAIN_MAX_EPOCHS} patience=${PRETRAIN_PATIENCE} batch=${PRETRAIN_BATCH_SIZE}" | tee -a "${MASTER_LOG}"
echo "[ablation] proto_tau=${PROTO_ASSIGN_TAU} proto_consistency_w=${PROTO_CONSISTENCY_W} proto_balance_w=${PROTO_BALANCE_W}" | tee -a "${MASTER_LOG}"
echo "[ablation] build_target_stage1=${BUILD_TARGET_STAGE1} target_stage1_dataset=${TARGET_STAGE1_DATASET} target_stage1_force=${TARGET_STAGE1_FORCE}" | tee -a "${MASTER_LOG}"
echo "[ablation] target_dataset=${TARGET_DATASET} pretrain_datasets=${PRETRAIN_DATASETS}" | tee -a "${MASTER_LOG}"
echo "[ablation] conda_env=${CONDA_ENV:-<none>}" | tee -a "${MASTER_LOG}"

for MASK in ${MASK_LIST}; do
    for ROUTE in ${ROUTE_LIST}; do
        RUN_ID="${BASE_TS}_raw9m${MASK}_${ROUTE}"
        RUN_DIR="./result/${RUN_ID}"
        LOG_FILE="${RUN_DIR}/ab_cora_raw9m${MASK}_${ROUTE}_${BASE_TS}.log"
        mkdir -p "${RUN_DIR}"

        PROTO_SOFT=0
        DOWNSTREAM_BANK=dataset
        if [[ "${ROUTE}" == "prototype" ]]; then
            PROTO_SOFT=1
            DOWNSTREAM_BANK=prototype
        fi

        echo "[ablation] start mask=${MASK} route=${ROUTE} run_id=${RUN_ID}" | tee -a "${MASTER_LOG}"
        echo "[ablation] log file: ${LOG_FILE}" | tee -a "${MASTER_LOG}"

        if [[ "${BUILD_TARGET_STAGE1}" == "1" ]]; then
          TARGET_STAGE1_CSV="${RUN_DIR}/stage_outputs/stage1/${TARGET_STAGE1_DATASET,,}/structural_encodings.csv"
          if [[ "${TARGET_STAGE1_FORCE}" == "1" || ! -f "${TARGET_STAGE1_CSV}" ]]; then
            echo "[ablation] build target stage1 for ${TARGET_STAGE1_DATASET}: ${TARGET_STAGE1_CSV}" | tee -a "${MASTER_LOG}"
            run_python ./src/train/run_stage_pipeline.py \
              --stage 1 \
              --datasets "${TARGET_STAGE1_DATASET}" \
              --data-root "./data" \
              --output-root "${RUN_DIR}/stage_outputs/stage1" \
              --max-subgraph-nodes 8 \
              --subgraph-hop-n 2 \
              --num-clusters 8 \
              --encoding-mode raw9 \
              --feature-mask "${MASK}" \
              --use-log1p 1 \
              --remove-selfloops 1 \
              --seed 39 \
              >> "${LOG_FILE}" 2>&1
          else
            echo "[ablation] reuse target stage1: ${TARGET_STAGE1_CSV}" | tee -a "${MASTER_LOG}"
          fi
        fi

        run_python ./src/train/execute.py \
          --dataset "${TARGET_DATASET}" \
          --result_subdir "${RUN_ID}" \
          --pretrain_datasets ${PRETRAIN_DATASETS} \
          --pretrain_method EGRAPHCL \
          --gpu 0 \
          --skip_pretrain 0 \
          --ablation_pre all \
          --ablation_down all \
          --force_rebuild_stage1 1 \
          --force_rebuild_stage2 1 \
          --force_rebuild_global_pool 1 \
          --rebuild_run_cache 1 \
          --stage1_num_clusters 64 \
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
          --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}" \
          --pretrain_max_epochs "${PRETRAIN_MAX_EPOCHS}" \
          --pretrain_patience "${PRETRAIN_PATIENCE}" \
          --pretrain_min_delta 1e-4 \
          --egraphcl_negative_pair_mode topk \
          --egraphcl_negative_topk 32 \
          --egraphcl_log_every 10 \
          --egraphcl_encoder_mode batched \
          --prompt_routing_mode "${ROUTE}" \
          --prototype_soft_prompt "${PROTO_SOFT}" \
          --prototype_assign_tau "${PROTO_ASSIGN_TAU}" \
          --prototype_consistency_loss_weight "${PROTO_CONSISTENCY_W}" \
          --prototype_balance_loss_weight "${PROTO_BALANCE_W}" \
          --downstream_prompt_bank_mode "${DOWNSTREAM_BANK}" \
          --max_subgraph_nodes 64 \
          --context_outputs_root "./result/${RUN_ID}/stage_outputs" \
          >> "${LOG_FILE}" 2>&1

        echo "[ablation] done mask=${MASK} route=${ROUTE}" | tee -a "${MASTER_LOG}"
        echo | tee -a "${MASTER_LOG}"
    done
done

echo "[ablation] all runs finished." | tee -a "${MASTER_LOG}"
' bash "${BASE_TS}" "${MASK_LIST}" "${CONDA_ENV}" "${PYTHON_BIN}" "${ROUTE_LIST}" "${PRETRAIN_MAX_EPOCHS}" "${PRETRAIN_PATIENCE}" "${PRETRAIN_BATCH_SIZE}" "${PROTO_ASSIGN_TAU}" "${PROTO_CONSISTENCY_W}" "${PROTO_BALANCE_W}" "${BUILD_TARGET_STAGE1}" "${TARGET_STAGE1_DATASET}" "${TARGET_STAGE1_FORCE}" "${TARGET_DATASET}" "${PRETRAIN_DATASETS}" \
> "./launcher_${BASE_TS}.log" 2>&1 &

echo "✅ 已后台启动（nohup），关终端不影响"
echo "📜 总日志: ./run_all_${BASE_TS}.log"
echo "📜 启动日志: ./launcher_${BASE_TS}.log"