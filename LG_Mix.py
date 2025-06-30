# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
from model import *
from data_loader import *
import torch.nn.functional as F
import time

torch.manual_seed(543)
np.random.seed(500)

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        '''

        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        '''
        self.net = models.resnet18(pretrained=False, num_classes=100).to(self.args.device)

        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights):
        self.net.load_state_dict(global_weights)
        net_before = copy.deepcopy(self.net)
        self.net.train()
        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        delta_w = {}
        for key in self.net.state_dict().keys():
            delta_w[key] = self.net.state_dict()[key] - net_before.state_dict()[key]

        return copy.deepcopy(self.net.state_dict()), delta_w

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
        return accuracy

class FedAvg(object):
    def __init__(self, clients, args):
        self.clients = clients
        self.args = args
        '''

        if self.args.dataset == 'Fashion':
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        else:
            print("coming soon")
        '''
        self.global_model = models.resnet18(pretrained=False, num_classes=100).to(self.args.device)

        self.clients_acc = [[] for _ in range(args.num_clients)]

    def avg_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg

    def estimate_trace(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'fc' in name and 'weight' in name:
                    return torch.norm(param).item()
        return 1.0

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

            global_weights_before = copy.deepcopy(self.global_model.state_dict())

            local_weights, local_deltas = [], []
            for client in self.clients:
                local_w, delta = client.train(global_weights_before)
                local_weights.append(local_w)
                local_deltas.append(delta)

            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            delta_global = {k: global_weights[k] - global_weights_before[k] for k in global_weights.keys()}

            for i, client in enumerate(self.clients):
                local_trace = self.estimate_trace(client.net)
                global_trace = self.estimate_trace(self.global_model)
                lam = local_trace / (local_trace + global_trace + 1e-8)

                new_weights = {}
                for k in global_weights.keys():
                    new_weights[k] = global_weights_before[k] + lam * local_deltas[i][k] + (1 - lam) * delta_global[k]
                client.net.load_state_dict(new_weights)

            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = client.test(client.net)
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Personalized Model Accuracy (avg): {np.mean(accuracies)}%')

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
        self.dataset = 'Cifar100' #####Fashion,Cifar10,Cifar100
        self.non_iid = 'Dirichlet' ###Dirichlet,Pathological
        self.dirichlet_alpha = 0.5
        self.num_shard = 50
        self.method = 'Cifar100_LG_Mix_0.5'

args = Args()
begin_time = time.time()

if args.non_iid == 'Dirichlet':
    train_loader, test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
elif args.non_iid == 'Pathological':
    train_loader, test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader, test_loader = create_iid_client_dataloaders(args.dataset, num_clients=args.all_clients, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
fedavg = FedAvg(clients, args)
fedavg.train()
end_time = time.time()
print(f"Running time: {end_time - begin_time} seconds")
