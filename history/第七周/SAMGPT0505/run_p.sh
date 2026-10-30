nohup python src/execute.py \
    --dataset Cora \
    --pretrain_datasets Citeseer Pubmed Photo Computers FacebookPagePage LastFMAsia \
    --use_daaa 1 \
    --lambda_vert_schedule 1 \
    --use_down_shared_proto 1 \
    --cdg_splits 100 \
    --skip_pretrain 1 \
    > diagnostics/cdg_daaa_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "DA-AA PID: $!"
