cd /home/ustc1958/lxy/graph/1/SAMGPT0508
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

nohup python ./src/execute.py \
    --graphcl_batch_size 2048 \
    --graphcl_nce_lambda 0.005 \
    --graphcl_nce_temperature 0.2 \
    --cluster_keep_ratio 1.0 \
    --cluster_quality_weight 0 \
    --denoise_anchor_weight 1 \
    --denoise_negative_mask 1 \
    --denoise_sim_threshold 0.6 \
    --denoise_min_negatives 16 \
    --denoise_weight_alpha 1.0 \
    --denoise_log_interval 50 \
    --denoise_visualize 1 \
    --domain_align_lambda 0.0 \
    > ./result/20260509_denoise_masked_nce_b2048.log 2>&1 &
