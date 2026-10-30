#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------
# Stage-1: structural encoding
# ---------------------------
python ./src/context_pipeline/run_stage1.py \
    --datasets Cora Citeseer Pubmed \
    --data-root ./data \
    --output-root ./src/context_pipeline/outputs \
    --max-subgraph-nodes 64 \
    --encoding-mode raw9 \
    --num-clusters 8

# ----------------------------------------------
# Stage-2: learnability scoring (per dataset)
# ----------------------------------------------
python ./src/context_pipeline/run_stage2.py \
    --input-csv ./src/context_pipeline/outputs/cora/structural_encodings.csv \
    --output-dir ./src/context_pipeline/outputs/cora/stage2

python ./src/context_pipeline/run_stage2.py \
    --input-csv ./src/context_pipeline/outputs/citeseer/structural_encodings.csv \
    --output-dir ./src/context_pipeline/outputs/citeseer/stage2

python ./src/context_pipeline/run_stage2.py \
    --input-csv ./src/context_pipeline/outputs/pubmed/structural_encodings.csv \
    --output-dir ./src/context_pipeline/outputs/pubmed/stage2

# ----------------------------------------------
# Execute: unified pretrain + downstream eval
# ----------------------------------------------
python ./src/execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --pretrain_method GRAPHCL \
    --gpu 0 \
    --skip_pretrain 0 \
    --stage2_top_ratio 0.3 \
    --stage2_max_subgraph_nodes 64 \
    --struct_max_subgraph_nodes 64 \
    --context_outputs_root ./src/context_pipeline/outputs