#!/usr/bin/bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPT0428

# ===================== 自动生成时间戳 =====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 带时间戳的日志文件夹 & 文件
LOG_DIR=./result/logs
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p ./result

PYTHON_BIN=${PYTHON_BIN:-/home/ustc1958/miniconda3/envs/graph/bin/python}

# ===================== 启动训练 =====================
nohup "$PYTHON_BIN" ./src/execute.py \
  --dataset Cora \
  --skip_pretrain 0 \
  --pretrain_method GRAPHCL \
  --use_gdcl_debias_in_pretrain 0 \
  --lr 0.0005 \
  --drop_percent 0.2 \
  --unify_dim 50 \
  --hid_units 128 \
  --layers_num 2 \
  > "$LOG_FILE" 2>&1 &

echo "============================================="
echo "训练已启动！PID = $!"
echo "日志文件（带时间戳）：$LOG_FILE"
echo "============================================="