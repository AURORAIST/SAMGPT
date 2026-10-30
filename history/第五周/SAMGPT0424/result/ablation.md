实验结果汇总
格式说明
CSV格式：pretrain_datasets, downstream_dataset, acc_mean, acc_std, microf_mean, microf_std, macrof_mean, macrof_std

实验1：筛选 vs 不筛选
实验	stage2_top_ratio	准确率
exp1_1_no_filter	1.0 (不筛选)	28.990 ± 9.459
exp1_2_filter_30	0.3	36.230 ± 10.426
exp1_3_filter_50	0.5	30.410 ± 9.118
结论：筛选有效！ratio=0.3 时效果最好，提升约 7.2%

实验2：筛选比例敏感性
实验	ratio	准确率
exp2_ratio_01	0.1	34.650 ± 10.176
exp2_ratio_02	0.2	30.950 ± 9.857
exp2_ratio_01	0.1	34.650
exp2_ratio_07	0.7	29.730 ± 10.087
结论：ratio=0.1 或 0.3 效果较好，过高或过低都不好

实验3：Learnability 各组件贡献
实验	alpha(Novelty)	beta(Coverage)	gamma(Redundancy)	准确率
exp3_1_novelty	1	0	0	26.320 ± 7.624
exp3_2_coverage	0	1	0	38.500 ± 10.826
exp3_3_redundancy	0	0	1	25.800 ± 7.420
exp5_full	1	1	1	35.390 ± 10.061
结论：

Coverage 是最重要的指标！单独使用就能达到 38.5%
Novelty 和 Redundancy 贡献较小
完整组合(35.4%)反而不如单独用 Coverage(38.5%)
实验4：Top-M 敏感性
实验	top_m	准确率
exp4_topm_1	1	28.100 ± 7.407
exp4_topm_3	3	36.360 ± 9.710
exp4_topm_10	10	28.650 ± 7.873
exp5_topm_5	5	35.420 ± 10.030
exp5_topm_20	20	34.400 ± 10.024
结论：top_m=3 或 5 效果最好

关键发现
筛选有效：ratio=0.3 比不筛选提升 7.2%
Coverage 是核心：alpha=1, beta=1, gamma=1 不如 beta=1 单独使用
最优配置：stage2_top_ratio=0.3, stage2_top_m_per_cluster=3, 只用 Coverage