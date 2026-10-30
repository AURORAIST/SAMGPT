cd /home/ustc1958/lxy/graph/1/SAMGPT0504
nohup python src/execute.py \
    --skip_pretrain 0 \
    --dataset Cora \
    --use_scp 1 \
    --struct_feat_dim 3 \
    --struct_prompt_residual 1 \
    --struct_prompt_scale 0.05 \
    > ./result/050406yuan.log 2>&1 &
