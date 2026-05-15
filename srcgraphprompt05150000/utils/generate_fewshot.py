import os
import torch
import random
from torch_geometric.datasets import Twitch, Flickr, FacebookPagePage, Coauthor
from dataset import *
from process import *
import scipy.sparse as sp
import numpy as np

# ------------------------------------------------------------
# 该脚本用于生成 few-shot 训练子集（按类别每类 K 个样本）。
# 输出两套数据：
# 1) node few-shot：只保存采样出的节点 idx 以及对应 labels
# 2) graph few-shot：在 node few-shot 基础上，为每个中心节点构造 2-hop 子图节点集合，
#    并保存 idx/batch/labels，方便后续以“子图 batch”形式训练。
#
# 目录结构（以 Cora 为例）：
#   ./data/fewshot_cora/{k}-shot_cora/{i}/idx.pt, labels.pt
#   ./data/fewshot_cora_graph/{k}-shot_cora/{i}/idx.pt, batch.pt, labels.pt
# 其中：
#   k=1..num_shots
#   i=0..num_samples-1（注意 create_folders 里建的是 1..num_samples，但实际保存用 enumerate 从 0 开始）
# ------------------------------------------------------------


def save_sample(data, path):
    """将一个 few-shot sample 保存到指定目录。

    Args:
        data: dict，至少包含
            - 'idx': Tensor，采样节点或子图节点索引
            - 'labels': Tensor，对应标签
            可选包含：
            - 'batch': Tensor，子图 batch 分组信息（graph few-shot 会用到）
        path: 保存目录
    """
    os.makedirs(path, exist_ok=True)
    torch.save(data['idx'], os.path.join(path, 'idx.pt'))
    torch.save(data['labels'], os.path.join(path, 'labels.pt'))
    if 'batch' in data:
        torch.save(data['batch'], os.path.join(path, 'batch.pt'))


def generate_fewshot_samples(labels, num_shots, num_samples):
    """按“每类 num_shots 个节点”的规则生成 num_samples 组 few-shot 采样。

    采样策略：
    - 遍历所有出现过的类别 label
    - 从该类样本索引中不放回随机抽取 num_shots 个
    - 将所有类别抽到的索引拼成一个样本的 idx

    注意：
    - 若某个类别的样本数 < num_shots，则该类别会被跳过（continue）。
      这会导致不同 shot 下样本实际包含的类别数可能不一致。

    Args:
        labels: Tensor [N]，节点标签
        num_shots: 每类采样数 K
        num_samples: 采样重复次数（生成多少个不同的 few-shot split）

    Returns:
        fewshot_samples: list[dict]
            每个 dict 包含：
            - 'idx': Tensor[总采样数]
            - 'labels': Tensor[总采样数]
    """
    fewshot_samples = []
    unique_labels = torch.unique(labels)
    for _ in range(num_samples):
        samples = []
        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) < num_shots:
                continue
            selected_indices = random.sample(label_indices.tolist(), num_shots)
            samples.extend(selected_indices)
        fewshot_samples.append({'idx': torch.tensor(samples), 'labels': labels[samples]})
    return fewshot_samples


def create_folders(base_path, dataset_name, num_shots=10, num_samples=100):
    """预创建保存目录。

    目录命名：{shot}-shot_{dataset}/{i}/

    注意：这里 i 从 1..num_samples，但后续保存时 enumerate(samples) 是从 0 开始，
    因此会多出一个未被使用的文件夹（以及遗漏最后一个），不过不影响读取（只要按实际保存路径读取）。
    """
    for shot in range(1, num_shots + 1):
        for i in range(1, num_samples + 1):
            os.makedirs(os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i)), exist_ok=True)


def save_fewshot_data(dataset_name, num_shots=10, num_samples=100, path='./data'):
    """生成并保存 node few-shot 数据（idx/labels）。"""
    print(f'Generating node_data for {dataset_name}')
    dataset = load_dataset(dataset_name, path)
    data = dataset[0]
    labels = data.y

    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples(labels, shot, num_samples)
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            save_sample(sample, sample_path)


def generate_fewshot_samples_graph(dataset_name, shotnum, num_samples, path='./data', subgraph_hop=2, max_neighbors=64):
    """在 node few-shot 的基础上，构造 graph few-shot（2-hop 子图 batch）。

    流程：
    1) 加载原始数据集（取 data[0]）
    2) 调用 process_tu 得到 (features, adj)
       - 注意：这里用 data['x'].shape[1] 作为 class_num 参数
         实际含义取决于 data.x 的组织方式
    3) 读取之前保存好的 node few-shot：idx_train / lbl_train
    4) 对 idx_train 调用 build_subgraph(adj, idx_train)：
       - 得到子图节点集合 subgraph['idx']
       - 以及对应 batch 分组 subgraph['batch']

    Returns:
        samples: list[dict]，每个 dict 包含 idx/batch/labels
    """
    data = load_dataset(dataset_name, path)[0]
    features, adj = process_tu(data, data['x'].shape[1])
    samples = []
    for i in range(num_samples):
        idx_train = torch.load(
            f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/idx.pt"
        ).type(torch.long)
        lbl_train = torch.load(
            f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/labels.pt"
        ).type(torch.long)

        subgraph = build_subgraph(adj, idx_train, subgraph_hop=subgraph_hop, max_neighbors=max_neighbors)
        samples.append({'idx': subgraph['idx'], 'batch': subgraph['batch'], 'labels': lbl_train})
    return samples



def save_fewshot_graph_data(dataset_name, num_shots=10, num_samples=100, subgraph_hop=2, max_neighbors=64):
    """生成并保存 graph few-shot 数据（idx/batch/labels）。"""
    print(f'Generating graph_data for {dataset_name}')
    path = './data'
    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}_graph')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples_graph(
            dataset_name,
            shot,
            num_samples,
            path,
            subgraph_hop=subgraph_hop,
            max_neighbors=max_neighbors,
        )
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            # batch = torch.arange(len(sample['idx']))
            # sample['batch'] = batch
            save_sample(sample, sample_path)


if __name__ == '__main__':
    # 直接运行会为列表内每个数据集生成 few-shot 数据
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'LastFMAsia', 'FacebookPagePage']
    # datasets =   ['LastFMAsia']
    for dataset_name in datasets:
        save_fewshot_data(dataset_name)
        save_fewshot_graph_data(dataset_name)
