# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import models
from model import *
from data_loader import *
import time

torch.manual_seed(543)

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.v_net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.optimizer = optim.SGD(self.v_net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model):
        self.v_net.train()
        global_model.eval()
        for epoch in range(self.args.local_epochs):
            # Mixed model: w_i = alpha * v_i + (1 - alpha) * w
            with torch.no_grad():
                for mp, gp in zip(self.v_net.parameters(), global_model.parameters()):
                    mp.data = self.args.alpha * mp.data + (1 - self.args.alpha) * gp.data
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.v_net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        # Return delta for global model update
        delta = copy.deepcopy(global_model.state_dict())
        for key in delta:
            delta[key] = self.v_net.state_dict()[key] - delta[key]
        return delta

    def test(self):
        self.v_net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs = self.v_net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

class APFL(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        if self.args.dataset == 'Fashion':
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        else:
            print("coming soon")
        self.clients_acc = [[] for _ in range(args.num_clients)]

    def update_global_model(self, deltas):
        global_state = self.global_model.state_dict()
        for key in global_state:
            if global_state[key].dtype not in [torch.float32, torch.float64]:
                continue  # 跳过非浮点数参数
            for delta in deltas:
                global_state[key] += delta[key] / len(deltas)
        self.global_model.load_state_dict(global_state)
        
    def avg_weights(self,w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg

    def save_results(self):
        summary_file = f'./results/{self.args.dataset}/{self.args.method}.csv'
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]
                csv_writer.writerow(all_accs)

    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round+1}/{self.args.comm_round} ---')
                
            deltas = []
            for client in self.clients:
                delta = client.train(self.global_model)
                deltas.append(delta)

            self.update_global_model(deltas)

            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = client.test()
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Mean Personalized Accuracy: {np.mean(accuracies)}%')

        self.save_results()

class Args:
    def __init__(self):
        self.comm_round = 100
        self.all_clients = 10
        self.num_clients = 10
        self.local_epochs = 1
        self.lr = 0.001
        self.batch_size = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Cifar100' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.1 #dirichlet coefficient /non-IID degree
        self.num_shard = 50 #The number of categories into which the data set is divided
        self.method = 'Cifar100_APFL_0.1'
        self.alpha = 0.4  # key parameter for APFL

args = Args()
begin_time = time.time()
if args.non_iid == 'Dirichlet':
    train_loader, test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
elif args.non_iid == 'Pathological':
    train_loader, test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader, test_loader = create_iid_client_dataloaders(args.dataset, num_clients=args.all_clients, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
apfl = APFL(clients, args)
apfl.train()
end_time = time.time()
print(f"Running time: {end_time - begin_time} seconds")
