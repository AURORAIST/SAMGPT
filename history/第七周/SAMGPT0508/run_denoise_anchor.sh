cd /home/ustc1958/lxy/graph/1/SAMGPT0508
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph

nohup python ./src/execute.py \
    --graphcl_batch_size 4096 \
    --graphcl_nce_lambda 0 \
    --graphcl_nce_temperature 0.2 \
    --cluster_keep_ratio 1.0 \
    --cluster_quality_weight 0 \
    --denoise_anchor_weight 1 \
    --denoise_negative_mask 0 \
    --denoise_weight_alpha 1.0 \
    --denoise_log_interval 50 \
    --denoise_visualize 1 \
    --domain_align_lambda 0.0 \
    > ./result/20260509_denoise_anchor_b4096.log 2>&1 &
