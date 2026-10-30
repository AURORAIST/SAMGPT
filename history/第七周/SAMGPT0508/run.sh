cd /home/ustc1958/lxy/graph/1/SAMGPT0508
source /home/ustc1958/miniconda3/etc/profile.d/conda.sh
conda activate graph
nohup python ./src/execute.py \
    --pretrain_datasets Pubmed Photo \
    --max_epochs 1 \
    --graphcl_stream_domains 0 \
    --graphcl_batch_size -1 \
    --debug_save_full_sim 1 \
    --debug_print_full_sim 0 \
    --denoise_log_interval 1 \
    --nce_diag_interval 1 \
    --debug_sim_subset_index_file /home/ustc1958/lxy/graph/1/SAMGPT0508/result/sim_dumps/model_node_GRAPHCL__Pubmed_Photo_1.0_1.0_all_all_50_256_0.001_gcn_gclbs-1_gclneg1_gclmodence_gclnce0.5_gclstream0_clkeep1.0_clnum10_clbalfixed_clquota0_denaw0_denm__he50a6fede5e1/epoch_00000_index_map_neighbor500_from_idx0.csv \
    > ./result/2026050901.log 2>&1 &
