# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
import csv
from torchvision import datasets, transforms
from torch.utils.data import Subset
from model import *
from non_IID import create_client_dataloaders

torch.manual_seed(543)

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        
        if self.args.dataset == 'Fashion':
            self.net = fashion_LeNet().to(self.args.device)
        else:
            self.net = cifar10_ResNet18().to(self.args.device)
        self.previous_local_model = None
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def fuse_weights(self, global_weights, local_weights, alpha):
        return {key: torch.lerp(global_weights[key].float(), local_weights[key].float(), alpha) for key in global_weights.keys()}

    def load_global_model(self, global_model_state_dict, alpha):
        global_model = copy.deepcopy(global_model_state_dict)
        if self.previous_local_model is not None:
            fused_weights = self.fuse_weights(global_model, self.previous_local_model, alpha)
            self.net.load_state_dict(fused_weights)
        else:
            self.net.load_state_dict(global_model)

    def train(self):
        self.net.train()
        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                global_outputs = self.net(inputs)
                loss = self.criterion(global_outputs, labels)
                loss.backward()
                self.optimizer.step()
        self.previous_local_model = copy.deepcopy(self.net.state_dict())
        return self.net.state_dict()

    def test(self, net):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs = net(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

class Server(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        if self.args.dataset == 'Fashion':
            self.global_model = fashion_LeNet().to(self.args.device)
        else:
            self.global_model = cifar10_ResNet18().to(self.args.device)
        self.clients_acc = [[] for _ in range(args.num_clients)]

    def avg_weights(self,w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg

    def save_results(self):
        summary_file = './csv/{}/{}/results.csv'.format(args.dataset, args.method)
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]
                csv_writer.writerow(all_accs)

    def train(self):
        #ssm_losses = []
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round+1}/{self.args.comm_round} ---')
            

            local_weights = [client.train() for client_id, client in enumerate(self.clients)]
            pre_acc = [client.test(client.net) for client in self.clients]
            
            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            
            ser_acc = [client.test(self.global_model) for client in self.clients]
            SSM_loss = np.array(pre_acc) - np.array(ser_acc)
            
            for client_id, client in enumerate(self.clients):
                ssm_loss = -SSM_loss[client_id]/100
                loss_alpha = np.exp(ssm_loss) / (1 + np.exp(ssm_loss))
                #print(loss_alpha)
                client.load_global_model(self.global_model.state_dict(), loss_alpha)

            accuracies = [client.test(client.net) for client in self.clients]
            for client_id, acc in enumerate(accuracies):
                self.clients_acc[client_id].append(acc)
            print(f'Average Personalized Model Test Accuracy: {np.mean(accuracies)}%')
            
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
        self.dataset = 'Cifar100' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological
        self.dirichlet_alpha = 0.5 #dirichlet系数/non-IID程度
        self.num_shard = 50 #数据集分成的类别份数,num_shards = num_clients * (类别数 / 每客户端的类别数量)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.method = 'Our'  #Local/FedAvg/Ditto/Our/FedALA

# 示例使用
args = Args()

if args.non_iid == 'Dirichlet':
    train_loader,test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader,test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)
clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
server = Server(clients, args)
server.train()
