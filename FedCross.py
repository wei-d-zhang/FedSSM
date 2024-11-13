import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
from torchvision import datasets, transforms,models
from torch.utils.data import Subset
from model import *
from data_loader import *
import torch.nn.functional as F
import time 

from model import *

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

    def train(self, param):
        self.net.load_state_dict(param)
        self.net.train()
        for epoch in range(self.args.local_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
                self.optimizer.zero_grad()
                outputs,_ = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
        return self.net.state_dict()

    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                outputs,_ = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy


class Server():
    def __init__(self,clients, args):
        self.clients = clients
        self.args = args
        model_mapping = {
            'Fashion': Fashion_ResNet18,
            'Cifar10': cifar10_ResNet18,
            'Cifar100': cifar100_ResNet18
        }
        self.model = model_mapping.get(self.args.dataset, lambda: None)().to(self.args.device)
        self.Middleware_model = [copy.deepcopy(self.model.state_dict()) for _ in range(int(self.args.client_num * self.args.active_client_rate))]

        
        self.clients_acc = [[] for i in range(args.client_num)]
        self.alpha = 0.5
        self.Is_Dynamic_alpha = False

    def run(self):
        for communication_round in range(self.args.comm_round):

            # select client
            if self.args.active_client_rate < 1:
                selected_client_ids = self.select_clients()
            else:
                selected_client_ids = np.arange(self.args.client_num)
            
            random_client_ids = copy.deepcopy(selected_client_ids)
            random.shuffle(random_client_ids)

            # Dynamic α-based acceleration
            if self.Is_Dynamic_alpha and self.alpha < 0.99:
                self.Dynamic_alpha()
                if self.alpha > 0.99:
                    self.alpha = 0.99

            # local train
            for client_index in selected_client_ids:
                now_param = self.clients[client_index].train(self.Middleware_model[random_client_ids[client_index]])
                self.Middleware_model[random_client_ids[client_index]] = now_param

            # update Middleware model
            self.CrossAggr()

           
            global_param = self.update_model(self.Middleware_model)
            self.model.load_state_dict(global_param)

            accuracies = []
            for client_id, client in enumerate(self.clients):
                acc = self.clients[client_id].test()
                self.clients_acc[client_id].append(acc)
                accuracies.append(acc)
            print(f'round{communication_round} Test Accuracy: {np.mean(accuracies)}%')
        print("Over")
        self.save_results()
        
    def save_results(self):
        summary_file = f'./csv/{self.args.dataset}/iid/{self.args.method}.csv'
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            for client_id in range(len(self.clients)):
                all_accs = self.clients_acc[client_id]
                csv_writer.writerow(all_accs)

    def Dynamic_alpha(self):
        # Dynamic α-based acceleration
        self.alpha = self.alpha + 0.005
    def select_clients(self):
        select_clients = np.random.choice(range(self.args.client_num), int(self.args.client_num * self.args.active_client_rate),
                                          replace=False)
        return select_clients

    def update_model(self, w):
        with torch.no_grad():
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def CrossAggr(self):
        w_list = copy.deepcopy(self.Middleware_model)
        for client_id in range(len(self.Middleware_model)):
            # Collaborative Model Selection
            select_client_id = self.CoModelSel_Min_Sim(client_id, w_list)
            # Cross-Aggregation
            for key in self.Middleware_model[client_id].keys():
                self.Middleware_model[client_id][key] = self.alpha * w_list[client_id][key] + (1 - self.alpha) * w_list[select_client_id][key]

    def CoModelSel_Min_Sim(self, client_id, w_list):
        sim_list = []
        for i in range(len(w_list)):
            if i != client_id:
                sim_list.append(self.calculateDistance_cosine(w_list[client_id], w_list[i]))
        min_sim_id = sim_list.index(min(sim_list))
        return min_sim_id

    def CoModelSel_Max_Sim(self, client_id, w_list):
        sim_list = []
        for i in range(len(w_list)):
            if i != client_id:
                sim_list.append(self.calculateDistance_cosine(w_list[client_id], w_list[i]))
        min_sim_id = sim_list.index(max(sim_list))
        return min_sim_id

    def calculateDistance_cosine(self, state_dict1, state_dict2):
        vector1 = self.state_to_vector(state_dict1)
        vecror2 = self.state_to_vector(state_dict2)

        cos_sim = torch.cosine_similarity(vector1.unsqueeze(0), vecror2.unsqueeze(0))
        return cos_sim

    def state_to_vector(self, state_dict):
        vector = torch.cat([par.flatten()for par in state_dict.values()])
        return vector

class Args:
    def __init__(self):
        self.comm_round = 100  # Communication rounds
        self.all_clients = 10  # Total number of clients
        self.client_num = 10  # The number of clients selected per round
        self.local_epochs = 1  # Number of client local training rounds
        self.lr = 0.001  # Learning rate
        self.batch_size = 64  # Batch size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = 'Cifar10' ####Fashion/Cifar10/Cifar100
        self.non_iid = 'Dirichlet' ##Dirichlet/Pathological/iid
        self.dirichlet_alpha = 0.1 #dirichlet coefficient /non-IID degree
        self.num_shard = 50 #The number of categories into which the data set is divided
        self.active_client_rate = 1
        self.method = 'FedCross_iid'  #FedAvg/Ditto/Our/FedALA/Local

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
server.run()

end_time = time.time()
run_time = end_time - begin_time
print(f"Running time: {run_time} seconds")
