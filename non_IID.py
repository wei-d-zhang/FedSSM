import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import numpy as np

#torch.manual_seed(543)
np.random.seed(500)

def create_client_dataloaders(dataname,num_clients, alpha, batch_size, test_ratio=0.2):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataname == 'Fashion':
        # 加载FashionMNIST数据集
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar10':
        # 加载Cifar10数据集
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        print("coming soon")
        return

    # 将训练集和测试集合并
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # 获取所有样本的标签和索引
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    # 将训练集的索引保留 [0, 60000)，测试集索引从 60000 开始
    test_labels_indices_offset = len(train_dataset)
    test_labels_indices = np.arange(len(test_labels)) + test_labels_indices_offset

    # 合并训练集和测试集的标签和索引
    labels = np.concatenate((train_labels, test_labels))
    indices = np.concatenate((np.arange(len(train_labels)), test_labels_indices))

    # 获取每个标签对应的样本索引
    unique_labels = np.unique(labels)
    label_indices = {label: indices[labels == label] for label in unique_labels}

    # 为每个客户端生成类别分布
    dirichlet_dist = np.random.dirichlet(alpha * np.ones(num_clients), size=len(unique_labels))

    # 为每个客户端分配数据
    client_datasets = []
    for client_idx in range(num_clients):
        client_data_indices = []
        for label_idx, (class_label, class_idx) in enumerate(label_indices.items()):
            # 根据Dirichlet分布分配该类别的样本
            num_samples_class = int(len(class_idx) * dirichlet_dist[label_idx][client_idx])
            client_data_indices.extend(np.random.choice(class_idx, num_samples_class, replace=False))
        client_datasets.append(Subset(combined_dataset, client_data_indices))

    # 划分每个客户端的数据为训练集和测试集
    client_train_datasets = []
    client_test_datasets = []
    for client_dataset in client_datasets:
        train_size = int((1 - test_ratio) * len(client_dataset))
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = random_split(client_dataset, [train_size, test_size])
        client_train_datasets.append(client_train_dataset)
        client_test_datasets.append(client_test_dataset)

    # 创建DataLoader
    client_train_loaders = [DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True) for client_train_dataset in client_train_datasets]
    client_test_loaders = [DataLoader(client_test_dataset, batch_size=batch_size, shuffle=False) for client_test_dataset in client_test_datasets]

    return client_train_loaders, client_test_loaders

def create_client_dataloaders_pathological(dataname, num_clients, num_shards, batch_size, test_ratio=0.2):
    """
    Pathological Non-IID 数据分配方法：每个客户端只接收一部分类别的数据，且类别之间不重叠。
    参数：
    - dataname: 数据集名称 ('Fashion' or 'Cifar')
    - num_clients: 客户端数量
    - num_shards: 数据集分成的类别份数，每个客户端将分配若干个份数（如两个类别）
    - batch_size: DataLoader 的 batch 大小
    - test_ratio: 测试数据占比，默认为 0.2（即 20%）
    返回值：
    - client_train_loaders: 每个客户端的训练数据加载器
    - client_test_loaders: 每个客户端的测试数据加载器
    """
    transform = transforms.Compose([transforms.ToTensor()])

    if dataname == 'Fashion':
        # 加载 FashionMNIST 数据集
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar10':
        # 加载 CIFAR10 数据集
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataname == 'Cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        print("coming soon")
        return
        

    # 获取数据集大小
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # 获取训练集和测试集的标签
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    # 获取所有数据的标签和索引
    test_labels_indices_offset = len(train_dataset)
    test_labels_indices = np.arange(len(test_labels)) + test_labels_indices_offset
    labels = np.concatenate((train_labels, test_labels))
    indices = np.concatenate((np.arange(len(train_labels)), test_labels_indices))

    # 排序标签以确保同一类别的样本在一起
    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]

    # 计算每个 shard 包含的样本数
    num_samples_per_shard = len(labels) // num_shards

    # 将数据划分为 num_shards 个 shard，并为每个客户端分配 shard
    client_datasets = [[] for _ in range(num_clients)]
    shard_indices = np.array_split(sorted_indices, num_shards)

    # 为每个客户端随机分配 shard，确保每个客户端只包含少数类别
    shards_per_client = num_shards // num_clients
    for client_idx in range(num_clients):
        assigned_shards = np.random.choice(num_shards, shards_per_client, replace=False)
        for shard_idx in assigned_shards:
            client_datasets[client_idx].extend(shard_indices[shard_idx])

    # 转换为子集并划分为训练集和测试集
    client_train_datasets = []
    client_test_datasets = []
    for client_data_indices in client_datasets:
        client_dataset = Subset(combined_dataset, client_data_indices)
        train_size = int((1 - test_ratio) * len(client_dataset))
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = random_split(client_dataset, [train_size, test_size])
        client_train_datasets.append(client_train_dataset)
        client_test_datasets.append(client_test_dataset)

    # 创建DataLoader
    client_train_loaders = [DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True) for client_train_dataset in client_train_datasets]
    client_test_loaders = [DataLoader(client_test_dataset, batch_size=batch_size, shuffle=False) for client_test_dataset in client_test_datasets]

    return client_train_loaders, client_test_loaders



