2026.5.13
未做的改动（你报告中“可能改变方法定义/实验设置”的部分，按你要求暂不动）：

增强视图复用原图 mask（aug_mask_policy=original_anchor）没加。
负采样排除范围（neg_exclude_scope）没加。
负采样刷新周期（neg_refresh_interval）没加。
负采样策略/比例的消融（intra_neg_ratio 系列）没改。
稀疏/CSR 形式的 ego list 读出未实现。
微批化多源域未实现（你说下次再改）。