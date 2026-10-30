cd /home/ustc1958/lxy/graph/1/SAMGPT0427
nohup python ./src/execute.py \
    --use_subgraph_cl 1 \
    --pretrain_method SUBGRAPH_CL \
    --n_clusters 7 \
    --num_subgraphs 100 \
    > ./result/20260427.log 2>&1 &
