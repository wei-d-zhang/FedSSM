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
        self.net = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_reg = args.lambda_reg  # Regularization parameter λ

    def train(self, global_net):
        self.net.train()
        global_net.eval()  # Global model is fixed during local training
        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                
                # Add regularization term: λ/2 * ||v_i - w||^2
                reg_loss = 0
                for local_param, global_param in zip(self.net.parameters(), global_net.parameters()):
                    reg_loss += torch.norm(local_param - global_param) ** 2
                reg_loss = (self.lambda_reg / 2) * reg_loss
                
                total_loss = loss + reg_loss
                total_loss.backward()
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
        return accuracy

class FedMoE(object):
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
        self.clients_acc = [[] for i in range(args.num_clients)]
        
    def avg_weights(self, w):
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
            
            for client in self.clients:
                client.net.load_state_dict(self.global_model.state_dict())

            local_weights = []
            for client in self.clients:
                local_w = client.train(self.global_model)
                local_weights.append(local_w)

            # Aggregate local models to update global model
            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            # Test each client's local model
            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = client.test(client.net)  # Test on local model
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'Global Model Test Accuracy: {np.mean(accuracies)}%')
            
        self.save_results()

class Args:
    def __init__(self):
        self.comm_round = 100  # Communication rounds
        self.all_clients = 10  # Total number of clients
        self.num_clients = 10  # The number of clients selected per round
        self.local_epochs = 1  # Number of client local training rounds
        self.lr = 0.001  # Learning rate
        self.batch_size = 64  # Batch size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Cifar100'  # Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet'  # Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.5  # Dirichlet coefficient / non-IID degree
        self.num_shard = 50  # The number of categories into which the data set is divided
        self.lambda_reg = 0.1  # Regularization parameter λ
        self.method = 'Cifar100_L2GD_0.5'  # FedAvg/Ditto/Our/FedALA/Local

args = Args()
begin_time = time.time()
if args.non_iid == 'Dirichlet':
    train_loader, test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
elif args.non_iid == 'Pathological':
    train_loader, test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader, test_loader = create_iid_client_dataloaders(args.dataset, num_clients=args.all_clients, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
fedmoe = FedMoE(clients, args)
fedmoe.train()
end_time = time.time()
run_time = end_time - begin_time
print(f"Running time: {run_time} seconds")