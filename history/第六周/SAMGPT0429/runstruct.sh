cd /home/ustc1958/lxy/graph/1/SAMGPT0429
nohup python src/execute.py \
    --dataset Cora \
    --pretrain_method GRAPHCL \
    --backbone gcn \
    --use_structure_conditioned_prompt 1 \
    --struct_feat_dim 3 \
    --struct_prompt_residual 1 \
    --struct_prompt_scale 0.05 \
    --skip_pretrain 0 \
    > ./result/2026050204.log 2>&1 &