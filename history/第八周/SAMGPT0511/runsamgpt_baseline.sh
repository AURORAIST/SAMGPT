#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ustc1958/lxy/graph/SAMGPT"
cd "$ROOT_DIR"

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

mkdir -p result

SEED=39
PRETRAIN_METHOD="GRAPHCL"
DATASET="Cora"
SHOT_NUM=1
HID_UNITS=256
LAYERS_NUM=3
UNIFY_DIM=50
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT_DIR/result/samgpt_baseline_${PRETRAIN_METHOD}_${DATASET}_seed${SEED}_shot${SHOT_NUM}_${TS}.log"

echo "[RUN] LOG_FILE=$LOG_FILE"
echo "[RUN] This runs the original SAMGPT baseline protocol from $ROOT_DIR/src/execute.py"
echo "[RUN] Protocol: last 100 test nodes, 100 few-shot splits, original downprompt, max 400 downstream steps."

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning:dgl.backend.pytorch.sparse"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
mkdir -p "$MPLCONFIGDIR"

{
echo "[RUN] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[RUN] ROOT_DIR=$ROOT_DIR"
echo "[RUN] pretrain_method=$PRETRAIN_METHOD dataset=$DATASET seed=$SEED shot_num=$SHOT_NUM hid_units=$HID_UNITS layers_num=$LAYERS_NUM unify_dim=$UNIFY_DIM"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 python -u src/execute.py \
  --pretrain_method "$PRETRAIN_METHOD" \
  --dataset "$DATASET" \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --shot_num "$SHOT_NUM" \
  --seed "$SEED" \
  --skip_pretrain 1 \
  --unify_dim "$UNIFY_DIM" \
  --hid_units "$HID_UNITS" \
  --layers_num "$LAYERS_NUM" \
  --lr 0.001 \
  --backbone gcn \
  --ablation_pre all \
  --ablation_down all \
  2>&1
} > "$LOG_FILE" &

echo "[RUN] started PID=$!"
echo "[RUN] tail -f $LOG_FILE"
echo "[RUN] grep -E 'Average accuracy|Mean:|Std :' $LOG_FILE"
