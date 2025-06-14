# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
import csv
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
from model import *
from data_loader import *
import time
import random
import logging

torch.manual_seed(543)

class Client(object):
    def __init__(self, local_trainloader, local_testloader, args):
        self.trainloader = local_trainloader
        self.testloader = local_testloader
        self.args = args
        self.net = self.get_model(self.args.dataset).to(self.args.device)
        self.previous_local_model = None
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
    def get_model(self,dataset):
        model = models.resnet18(pretrained=False)
        if dataset in ['Fashion', 'Cifar10']:
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif dataset == 'Cifar100':
            model.fc = nn.Linear(model.fc.in_features, 100)
        return model

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
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
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
        self.global_model = self.get_model(self.args.dataset, self.args.device)
        self.clients_acc = [[] for i in range(args.all_clients)]
        self.avg_accuracies = []
        os.makedirs(f'./results/{self.args.dataset}', exist_ok=True)
        self.summary_file = f'./results/{self.args.dataset}/{self.args.method}.csv'
        
    def get_model(self,dataset, device):
        model = models.resnet18(pretrained=False)  
        if dataset in ['Fashion', 'Cifar10']:
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif dataset == 'Cifar100':
            model.fc = nn.Linear(model.fc.in_features, 100)
        return model.to(device)

    def avg_weights(self, local_weights):
        if not local_weights:
            raise ValueError("Local weights list is empty.")
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key]
            avg_weights[key] = avg_weights[key].float() / len(local_weights)
        return avg_weights

    def save_results1(self):
        try:
            with open(self.summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [f'Client_{i}' for i in range(self.args.all_clients)] + ['Avg_Accuracy']
                writer.writerow(headers)
                max_rounds = max(len(acc_list) for acc_list in self.clients_acc)
                for round_idx in range(max_rounds):
                    row = [
                        self.clients_acc[i][round_idx] if round_idx < len(self.clients_acc[i]) else ''
                        for i in range(self.args.all_clients)
                    ]
                    row.append(self.avg_accuracies[round_idx] if round_idx < len(self.avg_accuracies) else '')
                    writer.writerow(row)
            logging.info(f"Results saved to {self.summary_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            
    def save_results(self):
        try:
            with open(self.summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [f'Client_{i}' for i in range(self.args.all_clients)] + ['Avg_Accuracy', 'Std_Dev']
                writer.writerow(headers)
                max_rounds = max(len(acc_list) for acc_list in self.clients_acc)
                for round_idx in range(max_rounds):
                    row = [
                        self.clients_acc[i][round_idx] if round_idx < len(self.clients_acc[i]) else ''
                        for i in range(self.args.all_clients)
                    ]
                    valid_acc = [self.clients_acc[i][round_idx] for i in range(self.args.all_clients)
                                 if round_idx < len(self.clients_acc[i]) and self.clients_acc[i][round_idx] != '']
                    avg = np.mean(valid_acc) if valid_acc else ''
                    std = np.std(valid_acc) if valid_acc else ''
                    row.append(avg)
                    row.append(std)
                    writer.writerow(row)
            logging.info(f"Results saved to {self.summary_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            
    def train(self):
        for comm_round in range(self.args.comm_round):
            print(f'\n--- Communication Round {comm_round+1}/{self.args.comm_round} ---')
            
            selected_clients = random.sample(self.clients, self.args.select_clients)
            local_weights = [client.train() for client in selected_clients]  # w_i^t
            
            selected_client_acc = {}
            for client in selected_clients:
                acc = client.test(client.net)
                selected_client_acc[client] = acc
            
            current_round_acc = []
            for client in self.clients:
                if client in selected_client_acc:
                    acc = selected_client_acc[client]
                    current_round_acc.append(acc)
                    self.clients_acc[self.clients.index(client)].append(acc)
                else:
                    current_round_acc.append(None)
                    self.clients_acc[self.clients.index(client)].append('')
            
            valid_acc = [acc for acc in current_round_acc if acc is not None]
            if valid_acc:
                avg_pre_acc = np.mean(valid_acc)
            else:
                avg_pre_acc = 0.0
            self.avg_accuracies.append(avg_pre_acc)
            print(f'Average Personalized Model Test Accuracy: {avg_pre_acc}%')
            
            global_weights = self.avg_weights(local_weights)    # w^t
            self.global_model.load_state_dict(global_weights)
            
            ser_acc = [client.test(self.global_model) for client in selected_clients]
            ssm_loss = np.array(list(selected_client_acc.values())) - np.array(ser_acc)
            
            for idx, client in enumerate(selected_clients):
                ssm_loss_value = -0.01 * ssm_loss[idx]
                loss_alpha = np.exp(ssm_loss_value) / (1 + np.exp(ssm_loss_value))
                client.load_global_model(self.global_model.state_dict(), loss_alpha)
            
        self.save_results()

class Args:
    def __init__(self):
        self.comm_round = 100  # Communication rounds
        self.all_clients = 100  # Total number of clients
        self.select_clients = 100  # The number of clients selected per round
        self.local_epochs = 1  # Number of client local training rounds
        self.lr = 0.001  # Learning rate
        self.batch_size = 64  # Batch size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Cifar10' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.5 #dirichlet coefficient /non-IID degree
        self.num_shard = 50 #The number of categories into which the data set is divided
        self.method = '%100_FedSSM'  #Local/FedAvg/Ditto/Our/FedALA

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

