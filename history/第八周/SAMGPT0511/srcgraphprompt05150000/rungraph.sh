#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ustc1958/lxy/graph/1/SAMGPT0511"
cd "$ROOT_DIR"

source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

mkdir -p result

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT_DIR/result/usp_subgraph_self_seed39_e20_down200_splits10_${TS}.log"

echo "[RUN] LOG_FILE=$LOG_FILE"

unset USP_NEG_DEBUG
export PYTHONWARNINGS="ignore::FutureWarning:dgl.backend.pytorch.sparse"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 nohup python srcgraphprompt/execute.py \
  --pretrain_method USP \
  --dataset Cora \
  --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
  --shots 1 \
  --seed 39 \
  --device cuda \
  --usp_epochs 20 \
  --usp_down_epochs 100 \
  --usp_eval_protocol samgpt_standard \
  --num_splits 100 \
  --split_start 0 \
  --unify_dim 50 \
  --hid_units 128 \
  --layers_num 2 \
  --subgraph_hop 2 \
  --readout prompt_weighted \
  --query_mode subgraph \
  --prototype attention \
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
  > "$LOG_FILE" 2>&1 &

echo "[RUN] started PID=$!"
echo "[RUN] tail -f $LOG_FILE"
echo "[RUN] grep 'USP result:' $LOG_FILE"