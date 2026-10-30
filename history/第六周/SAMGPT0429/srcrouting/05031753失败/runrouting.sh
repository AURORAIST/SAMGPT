# 1. 进入项目目录
cd /home/ustc1958/lxy/graph/1/SAMGPT0429

# 2. 创建日志目录（如已存在可跳过）
mkdir -p result

nohup python srcrouting/execute.py \
    --dataset Cora \
    --pretrain_method GRAPHCL \
    --backbone gcn \
    --use_structure_conditioned_prompt 1 \
    --struct_feat_dim 3 \
    --struct_prompt_residual 1 \
    --struct_prompt_scale 0.05 \
     --router_tau 3.0 \
    --random_struct 0 \
    --use_prompt_routing 0 \
    --freeze_generators 0 \
    --skip_pretrain 1 \
    > ./result/routing_3d09.log 2>&1 &
