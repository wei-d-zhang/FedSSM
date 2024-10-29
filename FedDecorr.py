# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import datasets, transforms,models
from torch.utils.data import Subset
from model import *
from non_IID import create_client_dataloaders,create_client_dataloaders_pathological
import torch.nn.functional as F

torch.manual_seed(543)

class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        
        # 初始化模型和优化器
        model_mapping = {
            'Fashion': fashion_LeNet,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, net):
        feddecorr = FedDecorrLoss()
        DECORR_COEF=0.1
        net.train()
        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
                self.optimizer.zero_grad()
                outputs,features = net(inputs)
                loss_feddecorr = feddecorr(features)
                
                loss = self.criterion(outputs, labels) + DECORR_COEF*loss_feddecorr
                loss.backward()
                self.optimizer.step()
                
        return self.net.state_dict()

    def test(self, net):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs,features = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        #print(f'Client Test Accuracy: {accuracy}%')
        return accuracy

class Server(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        if self.args.dataset == 'Fashion':
            self.global_model = fashion_LeNet().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        else:
            print("coming soon")
        self.clients_acc = [[] for i in range(args.num_clients)]
        
    def avg_weights(self,w):
        # 计算模型参数的加权平均
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg
    
    def save_results(self):
        # 定义汇总文件路径
        summary_file = f'./csv/{self.args.dataset}/{self.args.method}{self.args.non_iid}.csv'
        # 打开文件进行写操作
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            # 遍历所有客户端
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]      # 获取该客户端所有精度
                # 写入每个客户端的结果（客户端ID，最大精度，所有精度值）
                csv_writer.writerow(all_accs)

    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round+1}/{self.args.comm_round} ---')
            
            # 将全局模型分发给客户端
            for client in self.clients:
                client.net.load_state_dict(self.global_model.state_dict())

            # 客户端本地训练
            local_weights = []
            for client in self.clients:
                local_w = client.train(client.net)
                local_weights.append(local_w)

            # 服务器聚合模型
            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            # 测试全局模型在每个客户端上的性能
            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = self.clients[client_id].test(client.net)
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Global Model Test Accuracy: {np.mean(accuracies)}%')
            
        self.save_results()


# 定义参数
class Args:
    def __init__(self):
        self.comm_round = 100  # 通信轮数
        self.all_clients = 10  # 总客户端数
        self.num_clients = 10  # 每轮选择的客户端数
        self.local_epochs = 1  # 客户端本地训练轮数
        self.lr = 0.001  # 学习率
        self.batch_size = 64  # 批量大小
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Fashion' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Pathological' ##Dirichlet/Pathological
        self.dirichlet_alpha = 0.1 #dirichlet系数/non-IID程度
        self.num_shard = 50 #数据集分成的类别份数,num_shards = num_clients * (类别数 / 每客户端的类别数量)
        self.method = 'FedDecorr'  #FedAvg/Ditto/Our/FedALA/Local

# 示例使用
args = Args()
if args.non_iid == 'Dirichlet':
    train_loader,test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader,test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
server = Server(clients, args)
server.train()
