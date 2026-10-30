#!/bin/bash

# ============================================================
# Ablation Study: 子图筛选方法有效性验证
# ============================================================
# 后台运行实验，日志输出到文件
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
log "开始时间: $(date)"
log ""

# ============================================================
# 批次1：实验1 - 筛选 vs 不筛选
# ============================================================
log "=========================================="
log "批次1：筛选 vs 不筛选 (3个实验后台运行)"
log "=========================================="

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 1.0 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_1_no_filter" \
    > "$OUTPUT_DIR/exp1_1_no_filter.log" 2>&1 &

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.3 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_2_filter_30" \
    > "$OUTPUT_DIR/exp1_2_filter_30.log" 2>&1 &

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.5 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp1_3_filter_50" \
    > "$OUTPUT_DIR/exp1_3_filter_50.log" 2>&1 &

log "已启动3个后台任务，等待完成..."
wait
log "批次1完成"
log ""

# ============================================================
# 批次2：实验2 - 筛选比例敏感性
# ============================================================
log "=========================================="
log "批次2：筛选比例敏感性 (3个实验后台运行)"
log "=========================================="

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.1 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_01" \
    > "$OUTPUT_DIR/exp2_ratio_01.log" 2>&1 &

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.2 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_02" \
    > "$OUTPUT_DIR/exp2_ratio_02.log" 2>&1 &

nohup python src/train/execute.py \
    --dataset $DATASET \
    --pretrain_datasets $PRETRAIN_DATASETS \
    --downstream_task $DOWNSTREAM_TASK \
    --gpu $GPU \
    --seed $SEED \
    --stage2_top_ratio 0.7 \
    --skip_pretrain $SKIP_PRETRAIN \
    --pretrain_max_epochs $PRETRAIN_MAX_EPOCHS \
    --result_subdir "ablation/exp2_ratio_07" \
    > "$OUTPUT_DIR/exp2_ratio_07.log" 2>&1 &

log "已启动3个后台任务，等待完成..."
wait
log "批次2完成"
log ""

# ============================================================
# 批次3：实验3 - Learnability 各组件贡献
# ============================================================
log "=========================================="
log "批次3：Learnability 各组件贡献 (3个实验后台运行)"
log "=========================================="

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp3_1_novelty.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp3_2_coverage.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp3_3_redundancy.log" 2>&1 &

log "已启动3个后台任务，等待完成..."
wait
log "批次3完成"
log ""

# ============================================================
# 批次4：实验4 - Top-M 敏感性
# ============================================================
log "=========================================="
log "批次4：Top-M 敏感性 (3个实验后台运行)"
log "=========================================="

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp4_topm_1.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp4_topm_3.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp4_topm_10.log" 2>&1 &

log "已启动3个后台任务，等待完成..."
wait
log "批次4完成"
log ""

# ============================================================
# 批次5：剩余实验
# ============================================================
log "=========================================="
log "批次5：剩余实验 (3个实验后台运行)"
log "=========================================="

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp5_topm_5.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp5_topm_20.log" 2>&1 &

nohup python src/train/execute.py \
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
    > "$OUTPUT_DIR/exp5_full.log" 2>&1 &

log "已启动3个后台任务，等待完成..."
wait
log "批次5完成"
log ""

# ============================================================
# 结果汇总
# ============================================================
log "=========================================="
log "所有实验完成！"
log "结束时间: $(date)"
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