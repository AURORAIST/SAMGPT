cd /home/ustc1958/lxy/graph/1/SAMGPTcodebook || exit 1
# ensure result dir exists
mkdir -p result

# timestamped logfile for easier tracking
LOG=./result/$(date +%Y%m%d_%H%M%S)_cgpc.log

# run with unbuffered stdout/stderr for real-time logs
nohup python -u src/execute.py \
    --dataset Cora --use_cgpc \
    --lambda_context 0.05 --lambda_code 0.01 \
    --mask_type element --mask_rate 0.2 \
    --skip_pretrain 0 \
    --gpu 0 \
    --cgpc_num_codes 256 \
    --cgpc_tau_teacher 0.20 \
    --cgpc_tau_student 0.30 \
    --cgpc_sample_size 0 \
    --cgpc_uniform_reg_weight 2.0 \
    --cgpc_usage_uniform_weight 0.5 \
    --cgpc_ema_momentum 0.99 \
    --cgpc_pretrain_iters 20 \
    > "$LOG" 2>&1 &
echo "Started pid=$! log=$LOG"
