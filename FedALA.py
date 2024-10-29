import numpy as np
import torch
import torch.nn as nn
import copy
import csv
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from model import *
from non_IID import create_client_dataloaders
import random
from torch.utils.data import DataLoader
from typing import List, Tuple
from ALA import ALA


class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化模型和优化器
        self.net = fashion_LeNet().to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        # 初始化 ALA 参数
        self.ala = ALA(
            loss=self.criterion,
            train_data=self.trainloader,  # 直接使用数据集
            batch_size=self.args.batch_size,
            layer_idx=args.layer_idx,
            eta=args.lr,
            ALA_round = args.ala_round,
            device=self.device,
        )

    def adaptive_local_aggregation(self, global_model):
        self.ala.adaptive_local_aggregation(global_model, self.net)

    def train(self):
        self.net.train()
        for epoch in range(self.args.local_epochs):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            #print(f'Client Training - Epoch {epoch+1}, Loss: {running_loss/len(self.trainloader)}')
        return self.net.state_dict()

    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Client Test Accuracy: {accuracy}%')
        return accuracy

class Server(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        self.global_model = fashion_LeNet().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.clients_acc = [[] for _ in range(args.num_clients)]
        
    def avg_weights(self, w):
        # 计算模型参数的加权平均
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] /= len(w)
        return w_avg
    
    def save_results(self):
        for client_id in range(len(self.clients)):
            max_acc = max(self.clients_acc[client_id])
            client_test_file = './csv/{}/{}/c{}-{}.csv'.format(self.args.dataset,self.args.method, client_id + 1, max_acc)
            with open(client_test_file, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.clients_acc[client_id])

    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round + 1}/{self.args.comm_round} ---')
            
            # ALA自适应本地聚合
            for client in self.clients:
                client.adaptive_local_aggregation(self.global_model)

            # 客户端本地训练
            local_weights = []
            for client in self.clients:
                local_w = client.train()
                local_weights.append(local_w)

            # 服务器聚合模型
            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            # 测试全局模型在每个客户端上的性能
            accuracies = []
            for client_id in range(len(self.clients)):
                acc = self.clients[client_id].test()
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Global Model Test Accuracy: {np.mean(accuracies)}%')
            
# 定义参数
class Args:
    def __init__(self):
        self.comm_round = 10  # 通信轮数
        self.all_clients = 10  # 总客户端数
        self.num_clients = 10  # 每轮选择的客户端数
        self.local_epochs = 1  # 客户端本地训练轮数
        self.lr = 0.01  # 学习率
        self.batch_size = 64  # 批量大小
        self.layer_idx = 3
        self.dirichlet_alpha = 0.5 #dirichlet系数/non-IID程度
        self.dataset = 'Cifar' ####Fashion/Cifar
        self.ala_round = 10
        self.method = 'FedALA'  #FedAvg/FedProx/MambaFL/FedALA

# 示例使用
args = Args()

train_loader,test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
server = Server(clients, args)
server.train()
