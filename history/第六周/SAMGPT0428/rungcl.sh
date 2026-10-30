#!/usr/bin/bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT0428
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=./result/logs
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/ustc1958/miniconda3/envs/graph/bin/python

nohup "$PYTHON_BIN" ./srcgdcl/execute.py \
  --dataset Cora \
  --skip_pretrain 0 \
  --pretrain_method GRAPHCL \
  --use_gdcl_debias_in_pretrain 1 \
  --gdcl_debug_interval 100 \
  --gdcl_debias_temperature 1.0 \
  --gdcl_debias_lambda 0.1 \
  --lr 0.0005 \
  --drop_percent 0.2 \
  --unify_dim 50 \
  --hid_units 128 \
  --layers_num 2 \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG_FILE"