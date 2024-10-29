import torch
import random
import copy
import numpy as np
import time
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
import csv
from torchvision import datasets, transforms,models
from torch.utils.data import Subset
from model import fashion_LeNet,cifar10_ResNet18
from non_IID import create_client_dataloaders,create_client_dataloaders_pathological
import torch.nn.functional as F

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.mu = 0.1
        
        # 初始化模型和优化器
        model_mapping = {
            'Fashion': fashion_LeNet,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': lambda: models.resnet18(pretrained=False, num_classes=100)
        }
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_parameters):
        self.net.train()

        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                proximal_term = sum(((par - global_par) ** 2).sum() for par, global_par in zip(self.net.parameters(), global_parameters))
                loss = self.criterion(outputs, labels) + 0.5 * self.mu * proximal_term
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
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        #print(f'Client Test Accuracy: {accuracy}%')
        return accuracy


class Server():
    def __init__(self, clients, args):
        self.args = args
        self.clients = clients
        if self.args.dataset == 'Fashion':
            self.global_model = fashion_LeNet().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = models.resnet18(pretrained=False, num_classes=100).to(self.args.device)

        self.acc = []
        self.loss = []
        self.clients_acc = [[] for i in range(args.num_clients)]
  
        self.lam_momentum = 0.9  # momentum
        self.global_momentum = copy.deepcopy(self.global_model.state_dict())
        self.global_delta = copy.deepcopy(self.global_model.state_dict())
        
    def save_results(self):
        # 定义汇总文件路径
        summary_file = './csv/{}/{}.csv'.format(args.dataset, args.method)

        # 打开文件进行写操作
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)

            # 遍历所有客户端
            for client_id in range(len(self.clients)):
                max_acc = max(self.clients_acc[client_id])  # 找到该客户端的最大精度
                all_accs = self.clients_acc[client_id]      # 获取该客户端所有精度

                # 写入每个客户端的结果（客户端ID，最大精度，所有精度值）
                csv_writer.writerow(all_accs)

    def run(self):
        for communication_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {communication_round+1}/{self.args.comm_round} ---')

            delta_local = []
            w_local = []
            test_loss_sum = []
            test_acc_sum = []
            
            for client in self.clients:
                client.net.load_state_dict(self.global_model.state_dict())

            # FedACG lookahead momentum
            self.global_model.load_state_dict(self.fedACG_lookahead(copy.deepcopy(self.global_model)))
            global_state_dict = copy.deepcopy(self.global_model.state_dict())

            # Client
            for client in self.clients:
                now_param = client.train([copy.deepcopy(param) for param in self.global_model.parameters()])
                w_local.append(now_param)
                temp_delta = copy.deepcopy(now_param)
                for key in now_param:
                    temp_delta[key] = now_param[key] - global_state_dict[key]
                delta_local.append(temp_delta)

            # update global model delta
            global_param = self.aggregate(w_local, delta_local)
            self.global_model.load_state_dict(global_param)

            # test
            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = self.clients[client_id].test(self.global_model)
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Global Model Test Accuracy: {np.mean(accuracies)}%')
            
        print("Over")
        self.save_results()

    def aggregate(self, local_weights, local_deltas):
        # 计算经过训练后得到的全局模型，此时才是完整的全局模型，一个完整训练轮次结束
        global_weight = self.average(local_weights)

        # 更新动量，下发的时候使用
        self.global_delta = self.average(local_deltas)
        for param_key in self.global_momentum:
             self.global_momentum[param_key] = self.lam_momentum * self.global_momentum[param_key] + self.global_delta[param_key]
        return global_weight
    
    def average(self,weight_list):
        avg_weights = copy.deepcopy(weight_list[0])
        for param_key in avg_weights:
            for client_id in range(1, len(weight_list)):
                avg_weights[param_key] += weight_list[client_id][param_key]
            avg_weights[param_key] =avg_weights[param_key].float() / len(weight_list)
        return avg_weights

    @torch.no_grad()
    def fedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            sending_model_dict[key] = sending_model_dict[key].float() + self.lam_momentum * self.global_momentum[key].float()

        # model.load_state_dict(sending_model_dict)
        return copy.deepcopy(sending_model_dict)

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
        self.dataset = 'Cifar10' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Pathological' ##Dirichlet/Pathological
        self.dirichlet_alpha = 0.5 #dirichlet系数/non-IID程度
        self.num_shard = 50 #数据集分成的类别份数,num_shards = num_clients * (类别数 / 每客户端的类别数量)
        self.method = 'FedACG'  #FedAvg/Ditto/Our/FedALA/Local/FedPer/FedACG/ICFL

# 示例使用
args = Args()
if args.non_iid == 'Dirichlet':
    train_loader,test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader,test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
server = Server(clients, args)
server.run()
        