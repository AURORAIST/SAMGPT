cd /home/ustc1958/lxy/graph/1/SAMGPT
export PYTHONWARNINGS="ignore"
nohup python ./src/train/execute.py \
    --pretrain_method LP\
    > ./result/20260320LP.log 2>&1 &