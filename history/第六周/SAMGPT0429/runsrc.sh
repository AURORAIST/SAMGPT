cd /home/ustc1958/lxy/graph/1/SAMGPT0429 
    # --pretrain_datasets Citeseer \
nohup python srcnodeprotype/execute.py \
    --dataset Cora \
    --downstream_task node \
    --skip_pretrain 0 \
    --downstream_cluster_method kmeans \
    --enable_cluster_enhance 1 \
    --cluster_conf_threshold 0.2 \
    --cluster_interval 20 \
    --lambda_proto 0.2 \
    --unify_dim 128 \
    --nb_epochs 50 \
    > ./result/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &