cd /home/ustc1958/lxy/graph/1/SAMGPT0429
nohup python srcgoble/execute.py \
    --dataset Cora \
    --pretrain_method GRAPHCL \
    --backbone gcn \
    --use_structure_conditioned_prompt 1 \
    --struct_feat_dim 1 \
    --struct_prompt_residual 1 \
    --struct_prompt_scale 0.05 \
    --random_struct 0 \
    --skip_pretrain 1 \
    > ./result/goble1.log 2>&1 &