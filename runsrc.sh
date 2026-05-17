cd /home/ustc1958/lxy/graph/1/SAMGPT0516

mkdir -p result

TS=$(date +"%Y%m%d_%H%M%S")

nohup python ./src/execute.py \
    --skip_pretrain 0 \
    --reliability_loss 1 \
    --reliability_mode descriptor \
    --reliability_visualize 1 \
    --reliability_visual_interval 1000 \
    --reliability_log_interval 500 \
    > ./result/usp_reliability_${TS}.log 2>&1 &
