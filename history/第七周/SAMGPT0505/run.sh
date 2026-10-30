#!/usr/bin/env bash
# ============================================================
# run_daaa.sh — DA-AA-GPT 实验运行脚本
# ============================================================
set -euo pipefail

# ---- 可配置变量 ----
DATASET="${DATASET:-Cora}"
SHOT="${SHOT:-1}"
GPU="${GPU:-0}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
PRETRAIN_DATASETS="${PRETRAIN_DATASETS:-Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia}"

# DA-AA 超参
DAAA_N_SHARED="${DAAA_N_SHARED:-8}"
DAAA_N_DOMAIN="${DAAA_N_DOMAIN:-4}"
DAAA_LAMBDA_VERT="${DAAA_LAMBDA_VERT:-0.05}"
DAAA_LAMBDA_RES="${DAAA_LAMBDA_RES:-0.01}"
DAAA_LAMBDA_HSIC="${DAAA_LAMBDA_HSIC:-0.0}"
DAAA_LAMBDA_GC="${DAAA_LAMBDA_GC:-0.1}"

BASE_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="daaa_${DATASET}_${SHOT}shot_${BASE_TS}"
LOG_FILE="./diagnostics/${RUN_ID}.log"

# ✅ 关键修复：自动创建 result 目录
mkdir -p diagnostics

echo "============================================================"
echo "  DA-AA-GPT 实验 已后台启动"
echo "  数据集: ${DATASET}  Shot: ${SHOT}  GPU: ${GPU}"
echo "  运行ID: ${RUN_ID}"
echo "  日志: ${LOG_FILE}"
echo "============================================================"

# ✅ 真正完全后台（关闭终端不影响）
nohup python src/execute.py \
--dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method GRAPHCL \
    --gpu 0 --skip_pretrain 0 --shot_num 1 \
    --hid_units 256 --unify_dim 50 --layers_num 3 \
    --lr 0.001 --nb_epochs 10000 \
    --alpha 1.0 --beta 1.0 \
    --ablation_pre all --ablation_down all \
    --use_daaa \
    --skip_pretrain 1\
    --daaa_n_shared 8 --daaa_n_domain 8 \
    --daaa_lambda_vert 0.01 --daaa_lambda_res 0.1 \
    --daaa_lambda_hsic 0.001 --daaa_lambda_gc 0.1 \
    --max_k 24 --daaa_init_from_data \
    --daaa_monitor_interval 10 --daaa_vis_interval 200 \
    --daaa_expand_interval 200 --daaa_prune_interval 500 \
    > diagnostics/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "PID: $!"


echo "✅ 后台运行成功！"
echo "📊 实时查看日志：tail -f ${LOG_FILE}"
echo "============================================================"