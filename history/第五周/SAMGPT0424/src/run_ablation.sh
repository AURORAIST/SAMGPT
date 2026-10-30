#!/bin/bash

# ============================================================
# Ablation Study: 子图筛选方法有效性验证
# ============================================================
# 每次运行3个实验，等待完成后运行下一批
# ============================================================

# 基础配置
DATASET="Cora"
PRETRAIN_DATASETS="Cora Citeseer"
DOWNSTREAM_TASK="node"
SEED=39
SKIP_PRETRAIN=0
PRETRAIN_MAX_EPOCHS=100
GPU=0

# 输出目录
OUTPUT_DIR="result/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# 日志文件
LOG_FILE="$OUTPUT_DIR/ablation_log.txt"

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "Ablation Study: 子图筛选方法有效性验证"
log "=========================================="
log "输出目录: $OUTPUT_DIR"
log "基础配置: dataset=$DATASET, pretrain_datasets=$PRETRAIN_DATASETS"
log ""

# ============================================================
# 批次1：实验1 - 筛选 vs 不筛选
# ============================================================
log "=========================================="
log "批次1：筛选 vs 不筛选"
log "=========================================="

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 1.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_1_no_filter" \
    2>&1 | tee -a "$OUTPUT_DIR/exp1_1_no_filter.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_2_filter_30" \
    2>&1 | tee -a "$OUTPUT_DIR/exp1_2_filter_30.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.5 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_3_filter_50" \
    2>&1 | tee -a "$OUTPUT_DIR/exp1_3_filter_50.log" &

wait
log "批次1完成"
log ""

# ============================================================
# 批次2：实验2 - 筛选比例敏感性
# ============================================================
log "=========================================="
log "批次2：筛选比例敏感性"
log "=========================================="

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.1 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_01" \
    2>&1 | tee -a "$OUTPUT_DIR/exp2_ratio_01.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.2 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_02" \
    2>&1 | tee -a "$OUTPUT_DIR/exp2_ratio_02.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.7 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_07" \
    2>&1 | tee -a "$OUTPUT_DIR/exp2_ratio_07.log" &

wait
log "批次2完成"
log ""

# ============================================================
# 批次3：实验3 - Learnability 各组件贡献
# ============================================================
log "=========================================="
log "批次3：Learnability 各组件贡献"
log "=========================================="

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_alpha 1.0 \
    --stage2_beta 0.0 \
    --stage2_gamma 0.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp3_1_novelty" \
    2>&1 | tee -a "$OUTPUT_DIR/exp3_1_novelty.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_alpha 0.0 \
    --stage2_beta 1.0 \
    --stage2_gamma 0.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp3_2_coverage" \
    2>&1 | tee -a "$OUTPUT_DIR/exp3_2_coverage.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_alpha 0.0 \
    --stage2_beta 0.0 \
    --stage2_gamma 1.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp3_3_redundancy" \
    2>&1 | tee -a "$OUTPUT_DIR/exp3_3_redundancy.log" &

wait
log "批次3完成"
log ""

# ============================================================
# 批次4：实验4 - Top-M 敏感性
# ============================================================
log "=========================================="
log "批次4：Top-M 敏感性"
log "=========================================="

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_top_m_per_cluster 1 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp4_topm_1" \
    2>&1 | tee -a "$OUTPUT_DIR/exp4_topm_1.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_top_m_per_cluster 3 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp4_topm_3" \
    2>&1 | tee -a "$OUTPUT_DIR/exp4_topm_3.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_top_m_per_cluster 10 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp4_topm_10" \
    2>&1 | tee -a "$OUTPUT_DIR/exp4_topm_10.log" &

wait
log "批次4完成"
log ""

# ============================================================
# 批次5：剩余实验
# ============================================================
log "=========================================="
log "批次5：剩余实验"
log "=========================================="

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_top_m_per_cluster 5 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp5_topm_5" \
    2>&1 | tee -a "$OUTPUT_DIR/exp5_topm_5.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_top_m_per_cluster 20 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp5_topm_20" \
    2>&1 | tee -a "$OUTPUT_DIR/exp5_topm_20.log" &

python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --stage2_alpha 1.0 \
    --stage2_beta 1.0 \
    --stage2_gamma 1.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp5_full" \
    2>&1 | tee -a "$OUTPUT_DIR/exp5_full.log" &

wait
log "批次5完成"
log ""

# ============================================================
# 结果汇总
# ============================================================
log "=========================================="
log "所有实验完成！"
log "=========================================="
log "结果目录: $OUTPUT_DIR"
log ""

# 提取结果
echo "" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo "实验结果汇总" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

for dir in result/ablation/exp*; do
    if [ -d "$dir" ]; then
        csv_file=$(find $dir -name "*.csv" 2>/dev/null | head -1)
        if [ -n "$csv_file" ]; then
            echo "=== $(basename $dir) ===" | tee -a $LOG_FILE
            tail -3 $csv_file | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
        fi
    fi
done

log "Done!"