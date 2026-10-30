#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ustc1958/lxy/graph/1/SAMGPT0511"
cd "$ROOT_DIR"

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

mkdir -p result

SEED=39
USP_EPOCHS=0
USP_DOWN_EPOCHS=10
USP_DOWN_LOG_INTERVAL=2
NUM_SPLITS=100
SPLIT_START=0
QUERY_MODE="hybrid"
PROTOTYPE="attention"
READOUT="prompt_weighted"
EVAL_INTERVAL=0
DEVICE="cpu"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT_DIR/result/baseline_${QUERY_MODE}_${PROTOTYPE}_seed${SEED}_down${USP_DOWN_EPOCHS}_splits${NUM_SPLITS}_${TS}.log"

echo "[RUN] LOG_FILE=$LOG_FILE"

unset USP_NEG_DEBUG
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning:dgl.backend.pytorch.sparse"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
mkdir -p "$MPLCONFIGDIR"

{
echo "[RUN] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[RUN] ROOT_DIR=$ROOT_DIR"
echo "[RUN] baseline=epoch0 device=$DEVICE usp_epochs=$USP_EPOCHS usp_down_epochs=$USP_DOWN_EPOCHS usp_down_log_interval=$USP_DOWN_LOG_INTERVAL num_splits=$NUM_SPLITS query_mode=$QUERY_MODE prototype=$PROTOTYPE readout=$READOUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 python -u srcgraphprompt/execute.py \
  --pretrain_method USP \
  --dataset Cora \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --shots 1 \
  --seed "$SEED" \
  --device "$DEVICE" \
  --usp_epochs "$USP_EPOCHS" \
  --usp_down_epochs "$USP_DOWN_EPOCHS" \
  --usp_down_log_interval "$USP_DOWN_LOG_INTERVAL" \
  --downstream_eval_interval "$EVAL_INTERVAL" \
  --usp_eval_protocol samgpt_standard \
  --skip_pretrain 1 \
  --num_splits "$NUM_SPLITS" \
  --split_start "$SPLIT_START" \
  --unify_dim 50 \
  --hid_units 128 \
  --layers_num 2 \
  --subgraph_hop 2 \
  --readout "$READOUT" \
  --query_mode "$QUERY_MODE" \
  --prototype "$PROTOTYPE" \
  --negative_sampling domain_balanced \
  --neg_exclude_scope self \
  --neg_refresh_interval 5 \
  --num_negatives 16 \
  --intra_neg_ratio 0.5 \
  --ss_loss_mode sampled \
  --ss_num_negatives 128 \
  --aug_mask_policy original_anchor \
  --source_batch_size 2 \
  --csr_readout True \
  --align_type none \
  --progress_detail True \
  --time_profile True \
  2>&1
} > "$LOG_FILE" &

echo "[RUN] started PID=$!"
echo "[RUN] tail -f $LOG_FILE"
echo "[RUN] grep -E 'baseline_epoch0|USP result:|best_eval_acc' $LOG_FILE"
