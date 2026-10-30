import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import WebKB, Planetoid, Amazon, Coauthor, WikipediaNetwork, Reddit, \
    Flickr, PPI, Yelp, Twitch, Actor, KarateClub, FacebookPagePage, LastFMAsia
# BitcoinOTC

from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset
# from ogb.lsc import MAG240MDataset
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import numpy as np
import os
import shutil

# ------------------------------------------------------------
# 该文件用于：
# 1) 按名称加载不同来源的图节点分类数据集（PyG 内置 + OGB）
# 2) 对数据集做一些基本统计分析（度、聚类系数、最短路等）
# ------------------------------------------------------------

# 不同数据集 family 的名称列表，用于 load_dataset(name) 分发到对应的 PyG Dataset 类。
WebKB_datasets = ['Texas', 'Cornell', 'Wisconsin']
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo', 'Computers']
Coauthor_datasets = ['CS', 'Physics']
WikipediaNetwork_datasets = ['chameleon', 'squirrel']
Reddit_datasets = ['Reddit']
OGB_datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']

Flickr_datasets = ['Flickr']
PPI_datasets = ['PPI']
Yelp_datasets = ['Yelp']
Twitch_datasets = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
Actor_datasets = ['Actor']
KarateClub_datasets = ['KarateClub']
FacebookPagePage_datasets = ['FacebookPagePage']
LastFMAsia_datasets = ['LastFMAsia']
# BitcoinOTC_datasets = ['BitcoinOTC']
# MAG240MDatasets = ['MAG240MDataset']


def _safe_reload_planetoid(name: str, path: str):
    """更稳健地加载 Planetoid 数据集。

    背景：
    Planetoid 在不同 PyG 版本间 processed 文件格式可能不一致。
    某些情况下历史缓存会导致读取 processed 时触发诸如
    `ValueError: too many values to unpack` 的错误。

    策略：
    - 先尝试直接加载
    - 若捕获到特定解包错误：删除对应 processed 目录并触发重建

    Args:
        name: 数据集名称，如 'Cora'
        path: 数据根目录

    Returns:
        dataset: torch_geometric.datasets.Planetoid 实例
    """
    try:
        return Planetoid(root=path, name=name)
    except ValueError as e:
        msg = str(e)
        if 'too many values to unpack' not in msg:
            raise
        processed_dir = os.path.join(path, name, 'processed')
        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir, ignore_errors=True)
        return Planetoid(root=path, name=name)


def load_dataset(name, path='./data'):
    """按名称加载数据集。

    Args:
        name: 数据集名字（需出现在上面某个 *_datasets 列表里）
        path: 数据下载/缓存根目录

    Returns:
        dataset: PyG Dataset 对象

    Raises:
        ValueError: 未知数据集名称
    """
    if name in Planetoid_datasets:
        dataset = _safe_reload_planetoid(name=name, path=path)
    elif name in Amazon_datasets:
        dataset = Amazon(root=path, name=name)
    elif name in Coauthor_datasets:
        dataset = Coauthor(root=path, name=name)
    elif name in WebKB_datasets:
        dataset = WebKB(root=path, name=name)
    elif name in WikipediaNetwork_datasets:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in Reddit_datasets:
        dataset = Reddit(root=f'{path}/Reddit')
    elif name in OGB_datasets:
        dataset = PygNodePropPredDataset(root=path, name=name)
    elif name in Flickr_datasets:
        dataset = Flickr(root=f'{path}/Flickr')
    elif name in PPI_datasets:
        dataset = PPI(root=f'{path}/PPI')
    elif name in Yelp_datasets:
        dataset = Yelp(f'{path}/Yelp')
    elif name in Twitch_datasets:
        dataset = Twitch(root=path, name=name)
    elif name in Actor_datasets:
        dataset = Actor(root=f'{path}/Actor')
    elif name in KarateClub_datasets:
        dataset = KarateClub()
    elif name in FacebookPagePage_datasets:
        dataset = FacebookPagePage(root=f'{path}/Facebook')
    elif name in LastFMAsia_datasets:
        dataset = LastFMAsia(root=f'{path}/LastFMAsia')
    # elif name in BitcoinOTC_datasets:
    #     dataset = BitcoinOTC(root=f'{path}/BitcoinOTC')
    # elif name in MAG240MDatasets:
    #     dataset = MAG240MDataset(root=f'{path}/MAG240MDataset')
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # print(f'{name}: {dataset[0].num_nodes}')
    return dataset


def analyze_dataset(name, graph=False):
    """对单图数据集做基本统计与可选度分布可视化。

    说明：
    - 大多数节点分类基准（Planetoid/Amazon/WebKB/...）都是“单张大图”，因此取 dataset[0]。
    - 指标计算使用 NetworkX：需要把 PyG Data 转成 networkx.Graph。
      对于大图（如 Reddit/ogbn-products 等）全对最短路径会非常慢且占用巨大内存。

    Args:
        name: 数据集名
        graph: 是否绘制度分布直方图
    """
    data_ = load_dataset(name)
    # print(len(data_))
    data = data_[0]

    # 节点度（使用 edge_index 的 source 端计算无向度时一般需确保图已对称/或 to_undirected）
    deg = degree(data.edge_index[0], data.num_nodes)
    average_degree = deg.mean().item()

    print(
        f'\n{name}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, '
        f'num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}'
    )

    # 转为无向 NetworkX 图（便于计算聚类系数/最短路等图指标）
    G = to_networkx(data, to_undirected=True)

    # -------- 聚类系数（Clustering）--------
    print('Clustering Coefficient:')
    # transitivity = 3*三角形数 / 三元组数（全局聚类系数的一种定义）
    global_clustering_coefficient = nx.transitivity(G)
    print(f'Global Clustering Coefficient: {global_clustering_coefficient:.4f}')

    # 平均局部聚类系数
    average_clustering_coefficient = nx.average_clustering(G)
    print(f'Average Clustering Coefficient: {average_clustering_coefficient:.4f}')

    # -------- 最短路相关指标 --------
    # all_pairs_shortest_path_length 在大图上复杂度极高（近似 O(N*(N+E))），请谨慎使用。
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    path_lengths = []
    for node, lengths in shortest_path_lengths.items():
        path_lengths.extend(lengths.values())

    # 直径 = 最长最短路
    network_diameter = max(path_lengths)
    print(f'Network Diameter:')

    # 平均最短路长度
    average_shortest_path_length = np.mean(path_lengths)
    print(f'Average Shortest Path Length: {average_shortest_path_length:.4f}')

    # 90% 分位最短路长度
    percentile_90_shortest_path_length = np.percentile(path_lengths, 90)
    print(f'90th Percentile Shortest Path Length: {percentile_90_shortest_path_length:.4f}')

    # 可选：绘制度分布
    if graph:
        plt.figure()
        plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
        plt.title(f"Degree Distribution of {name}")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()


def analyze_dataset_multi(name, graph=False):
    """对多图数据集逐图分析（如 PPI 这类由多张图组成的数据集）。

    Args:
        name: 数据集名
        graph: 是否绘制度分布图
    """
    data_ = load_dataset(name)
    # print(len(data_))
    for i, data in enumerate(data_):
        deg = degree(data.edge_index[0], data.num_nodes)
        average_degree = deg.mean().item()
        print(
            f'{name}_{i+1}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, '
            f'num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}'
        )
        if graph:
            plt.figure()
            plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
            plt.title(f"Degree Distribution of {name}")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.show()


if __name__ == '__main__':
    # 直接运行本文件时，会对 selected_datasets 做统计输出
    selected_datasets = [
        # 'Texas', 'Cornell', 'Wisconsin',
        # 'Cornell',
        'Cora', 'Citeseer', 'Pubmed',
        'Photo',
        'Computers',
        # 'CS', 'Physics',
        # 'chameleon', 'squirrel',
        # 'Reddit',
        # 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', #'ogbn-mag',
        # 'Flickr',
        # 'PPI',
        # 'Yelp',
        # 'Actor',
        # 'ES',
        # 'DE', 'EN', 'ES',
        # 'FR', 'PT', 'RU',
        # 'KarateClub',
        'FacebookPagePage', 'LastFMAsia',
        # #'BitcoinOTC'
        # 'MAG240MDataset'
    ]

    for dataset in selected_datasets:
        analyze_dataset(dataset)
        # analyze_dataset_multi(dataset)