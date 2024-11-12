import numpy as np
import torch
import torch.nn as nn
import copy
import csv
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from model import *
from data_loader import *
import random
from torch.utils.data import DataLoader
from typing import List, Tuple
from ALA import ALA
import time


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
        self.ala = ALA(
            loss=self.criterion,
            train_data=self.trainloader,
            batch_size=self.args.batch_size,
            layer_idx=args.layer_idx,
            eta=args.lr,
            ALA_round = args.ala_round,
            device=args.device,
        )

    def adaptive_local_aggregation(self, global_model):
        self.ala.adaptive_local_aggregation(global_model, self.net)

    def train(self):
        self.net.train()
        for epoch in range(self.args.local_epochs):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                
                self.optimizer.zero_grad()
                outputs,_ = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
        return self.net.state_dict()

    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs,_ = self.net(inputs)
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
            self.global_model = Fashion_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar10':
            self.global_model = cifar10_ResNet18().to(self.args.device)
        elif self.args.dataset == 'Cifar100':
            self.global_model = cifar100_ResNet18().to(self.args.device)
        else:
            print("coming soon")
        self.clients_acc = [[] for _ in range(args.num_clients)]
        
    def avg_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key].float() / len(w)
        return w_avg
    
    def save_results(self):
        summary_file = f'./csv/{self.args.dataset}/iid/{self.args.method}.csv'
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]
                csv_writer.writerow(all_accs)

    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round + 1}/{self.args.comm_round} ---')
            
            for client in self.clients:
                client.adaptive_local_aggregation(self.global_model)

            local_weights = []
            for client in self.clients:
                local_w = client.train()
                local_weights.append(local_w)

            global_weights = self.avg_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            accuracies = []
            for client_id in range(len(self.clients)):
                acc = self.clients[client_id].test()
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
        self.dataset = 'Cifar10' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.1 #dirichlet coefficient /non-IID degree
        self.num_shard = 50 #The number of categories into which the data set is divided
        self.layer_idx = 3
        self.ala_round = 2
        self.method = 'FedALA_iid'  #FedAvg/Ditto/Our/FedALA/Local

args = Args()
begin_time = time.time()
if args.non_iid == 'Dirichlet':
    train_loader,test_loader = create_client_dataloaders(args.dataset, num_clients=args.all_clients, alpha=args.dirichlet_alpha, batch_size=args.batch_size, test_ratio=0.2)
elif args.non_iid == 'Pathological':
    train_loader,test_loader = create_client_dataloaders_pathological(args.dataset, num_clients=args.all_clients, num_shards=args.num_shard, batch_size=args.batch_size, test_ratio=0.2)
else:
    train_loader,test_loader =  create_iid_client_dataloaders(args.dataset, num_clients=args.all_clients, batch_size=args.batch_size, test_ratio=0.2)

clients = [Client(train_loader[i], test_loader[i], args) for i in range(args.all_clients)]
server = Server(clients, args)
server.train()

end_time = time.time()
run_time = end_time - begin_time
print(f"Running time: {run_time} seconds")
