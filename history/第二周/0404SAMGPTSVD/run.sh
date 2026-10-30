#!/usr/bin/env bash
set -euo pipefail

cd /home/ustc1958/lxy/graph/1/SAMGPTSVD
export PYTHONWARNINGS="ignore"

PYTHON_BIN="/home/ustc1958/miniconda3/envs/graph/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

RUN_SUBDIR="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="./result/${RUN_SUBDIR}"
LOG_FILE="${RUN_DIR}/run.log"
NOHUP_LOG_FILE="${RUN_DIR}/nohup.log"
mkdir -p "$RUN_DIR"
exec > >(tee "$LOG_FILE") 2>&1
echo "[run] log file: $LOG_FILE"
echo "[run] result dir: $RUN_DIR"
echo "[run] nohup log file: $NOHUP_LOG_FILE"


nohup "$PYTHON_BIN" ./src/train/execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method EGRAPHCL \
    --downstream_task node \
    --gpu 0 \
    --skip_pretrain 0 \
    --stage1_num_clusters 8 \
    --stage2_top_m_per_cluster 8 \
    --stage2_top_ratio 0.3 \
    --global_pool_svd_rank 9 \
    --lr 0.0003 \
    --egraphcl_tau 0.99 \
    --egraphcl_temperature 0.2 \
    --egraphcl_batch_size 128 \
    --max_subgraph_nodes 64 \
    --force_rebuild_global_pool 1 \
    --result_subdir "$RUN_SUBDIR" \
    --context_outputs_root "./result/${RUN_SUBDIR}/stage_outputs" \
    > "$NOHUP_LOG_FILE" 2>&1 &

echo "[run] started in background, pid=$!"