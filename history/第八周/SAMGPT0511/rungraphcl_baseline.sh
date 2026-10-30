#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ustc1958/lxy/graph/1/SAMGPT0511"
cd "$ROOT_DIR"

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

mkdir -p result

SEED=39
GRAPHCL_EPOCHS=2000
GRAPHCL_DOWN_EPOCHS=400
NUM_SPLITS=100
SPLIT_START=0
EVAL_INTERVAL=0
HID_UNITS=128
LAYERS_NUM=2
UNIFY_DIM=50
SKIP_PRETRAIN=0

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT_DIR/result/graphcl_baseline_seed${SEED}_e${GRAPHCL_EPOCHS}_down${GRAPHCL_DOWN_EPOCHS}_splits${NUM_SPLITS}_${TS}.log"

echo "[RUN] LOG_FILE=$LOG_FILE"
echo "[RUN] This is the normal GraphCL baseline for our method's node task protocol."
echo "[RUN] Protocol: Cora node, last 100 test nodes, 100 few-shot splits, transductive full target graph."

unset USP_NEG_DEBUG
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning:dgl.backend.pytorch.sparse"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
mkdir -p "$MPLCONFIGDIR"

{
  echo "[RUN] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[RUN] ROOT_DIR=$ROOT_DIR"
  echo "[RUN] pretrain_method=GRAPHCL graphcl_epochs=$GRAPHCL_EPOCHS graphcl_down_epochs=$GRAPHCL_DOWN_EPOCHS num_splits=$NUM_SPLITS hid_units=$HID_UNITS layers_num=$LAYERS_NUM unify_dim=$UNIFY_DIM skip_pretrain=$SKIP_PRETRAIN"
} > "$LOG_FILE"

nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
  PYTHONWARNINGS="$PYTHONWARNINGS" \
  MPLCONFIGDIR="$MPLCONFIGDIR" \
  python -u srcgraphprompt/execute.py \
  --pretrain_method GRAPHCL \
  --dataset Cora \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --shots 1 \
  --seed "$SEED" \
  --device cuda \
  --usp_epochs "$GRAPHCL_EPOCHS" \
  --usp_down_epochs "$GRAPHCL_DOWN_EPOCHS" \
  --downstream_eval_interval "$EVAL_INTERVAL" \
  --usp_eval_protocol samgpt_standard \
  --skip_pretrain "$SKIP_PRETRAIN" \
  --num_splits "$NUM_SPLITS" \
  --split_start "$SPLIT_START" \
  --unify_dim "$UNIFY_DIM" \
  --hid_units "$HID_UNITS" \
  --layers_num "$LAYERS_NUM" \
  --aug_type edge \
  --drop_percent 0.1 \
  --lr 0.001 \
  --backbone gcn \
  --ablation_pre all \
  --ablation_down all \
  --time_profile True \
  >> "$LOG_FILE" 2>&1 &

echo "[RUN] started PID=$!"
echo "[RUN] tail -f $LOG_FILE"
echo "[RUN] grep -E 'Loss:|Average accuracy|Mean:|Std :' $LOG_FILE"
