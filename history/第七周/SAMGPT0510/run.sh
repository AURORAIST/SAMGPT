cd /home/ustc1958/lxy/graph/1/SAMGPT0510
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph
nohup python ./src/execute.py \
    --skip_pretrain 0 \
    --pretrain_method GRAPHCL \
    --negative_sampler structural_hard \
    --structure_hops 3 \
    --structure_min_sim 0.05 \
    --structure_max_sim 0.5\
    --structure_candidates 64 \
    --force_rebuild_aug 1 \
    > ./result/2026050902.log 2>&1 &
