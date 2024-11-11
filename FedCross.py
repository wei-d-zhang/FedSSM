import torch
import random
import copy
import numpy as np
import time

from algorithm.FedCross.FedCrossClient import FedCross_client_model
from models.model import fashion_LeNet, FMNIIST_ResNet18, cifar10_LeNet, cifar10_ResNet18, Scrap_ResNet18


class Fed_Cross():
    def __init__(self, args, client_dataloaders):
        self.args = args
        self.client_dataloaders = client_dataloaders
        self.model = eval(args.model)().to(args.device)
        self.client_models = [FedCross_client_model(args=self.args, dataloader=self.client_dataloaders[i], id=i) for i in range(self.args.client_num)]
        self.Middleware_model = [copy.deepcopy(self.model.state_dict()) for _ in range(int(self.args.client_num * self.args.active_client_rate))]

        self.acc = []
        self.loss = []
        self.acc_clients = [[] for _ in range(self.args.client_num)]
        self.loss_clients = [[] for i in range(self.args.client_num)]

        self.old_client_models = [copy.deepcopy(self.client_models[i].model) for i in range(self.args.client_num)]
        self.now_client_models = [copy.deepcopy(self.client_models[i].model) for i in range(self.args.client_num)]
        self.count_layer_conflicts_list = []

        self.alpha = 0.5
        self.Is_Dynamic_alpha = False

    def run(self):

        # random_client_ids = np.arange(self.args.client_num)
        for communication_round in range(self.args.global_rounds):

            # select client
            if self.args.active_client_rate < 1:
                selected_client_ids = self.select_clients()
            else:
                selected_client_ids = np.arange(self.args.client_num)

            # Shuffle Middleware_model
            # if communication_round == 0:
            random_client_ids = copy.deepcopy(selected_client_ids)
            random.shuffle(random_client_ids)

            start_time = time.time()
            test_loss_sum = []
            test_acc_sum = []

            # Dynamic α-based acceleration
            if self.Is_Dynamic_alpha and self.alpha < 0.99:
                self.Dynamic_alpha()
                if self.alpha > 0.99:
                    self.alpha = 0.99

            # local train
            for client_index in selected_client_ids:
                # 保存上一轮模型参数
                # self.old_client_models[client_index].load_state_dict(
                    # self.client_models[client_index].model.state_dict())

                # now_param = self.client_models[client_index].train()
                now_param = self.client_models[client_index].train(self.Middleware_model[random_client_ids[client_index]])
                # 保存本地训练后模型参数
                # self.now_client_models[client_index].load_state_dict(now_param)
                # upload
                self.Middleware_model[random_client_ids[client_index]] = now_param

            # update Middleware model
            self.CrossAggr()

            # random_client_ids = copy.deepcopy(selected_client_ids)
            # random.shuffle(random_client_ids)

            # update client model
            # for client_index in selected_client_ids:
            #     self.client_models[client_index].model.load_state_dict(self.Middleware_model[random_client_ids[client_index]])

            # update global model
            global_param = self.update_model(self.Middleware_model)
            self.model.load_state_dict(global_param)

            # 计算模型更新方向与伪梯度方向不一致的数量
            # count_layer_conflicts = self.count_update_conflict()
            # self.count_layer_conflicts_list.append(count_layer_conflicts)
            # print(count_layer_conflicts)

            # test
            for client_index in selected_client_ids:
                loss_i, acc_i = self.client_models[client_index].test(self.model.state_dict())
                test_acc_sum.append(acc_i)
                test_loss_sum.append(loss_i)
                self.acc_clients[client_index].append(acc_i)
                self.loss_clients[client_index].append(loss_i)

            now_acc = sum(test_acc_sum)/len(test_acc_sum)
            now_loss = sum(test_loss_sum)/len(test_loss_sum)

            self.acc.append(round(now_acc, 2))
            self.loss.append(round(now_loss, 2))

            print("The loss of the {} round is :{} and the acc is:{},time cost: {}".format(communication_round, now_loss,
                                                                                         now_acc,
                                                                                         time.time() - start_time))
        print("Over")

        # avg_count_layer_conflicts = [0 for _ in range(int(self.args.model_layer_num / 2))]
        # for layer in range(int(self.args.model_layer_num / 2)):
        #     for i in range(len(self.count_layer_conflicts_list)):
        #         avg_count_layer_conflicts[layer] += self.count_layer_conflicts_list[i][layer]
        # print(avg_count_layer_conflicts)

        return self.loss_clients, self.acc, self.acc_clients

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

    def CoModelSel_Order(self):
        pass

    def calculateDistance_cosine(self, state_dict1, state_dict2):
        """
        Calculate the distance between model parameters
        """
        # 计算余弦相似度
        vector1 = self.state_to_vector(state_dict1)
        vecror2 = self.state_to_vector(state_dict2)

        cos_sim = torch.cosine_similarity(vector1.unsqueeze(0), vecror2.unsqueeze(0))
        return cos_sim

    def state_to_vector(self, state_dict):
        # 将state_dict转为一维向量
        vector = torch.cat([par.flatten()for par in state_dict.values()])
        return vector

    def count_update_conflict(self):
        count_layer_conflicts = [0 for _ in range(int(self.args.model_layer_num / 2))]
        for index in range(self.args.client_num):
            temp = self.calculate_update_conflict(index)
            for i in range(len(temp)):
                count_layer_conflicts[i] += temp[i]

        return count_layer_conflicts

    def calculate_update_conflict(self, index):
        # 只能用于lenet,mlp
        pseudo_gradient = []
        model_update_direction = []
        model_state = copy.deepcopy(self.model.state_dict())
        count_layer_conflict = []
        for key in model_state.keys():
            pseudo_gradient.append(self.now_client_models[index].state_dict()[key].cpu().flatten().detach().numpy() -
                                   self.old_client_models[index].state_dict()[key].cpu().flatten().detach().numpy())
            model_update_direction.append(
                self.client_models[index].model.state_dict()[key].cpu().flatten().detach().numpy() -
                self.now_client_models[index].state_dict()[key].cpu().flatten().detach().numpy())

        for i in range(0, self.args.model_layer_num, 2):
            temp_1 = np.concatenate((pseudo_gradient[i], pseudo_gradient[i + 1]))  # weight + bias
            temp_2 = np.concatenate((model_update_direction[i], model_update_direction[i + 1]))

            dot_product = np.dot(temp_1, temp_2)  # 计算向量点积
            norm_vec1 = np.linalg.norm(temp_1)  # 计算vec1的范数
            norm_vec2 = np.linalg.norm(temp_2)  # 计算vec2的范数
            if np.isnan(dot_product / (norm_vec1 * norm_vec2)):
                count_layer_conflict.append(0)
            else:
                count_layer_conflict.append(dot_product / (norm_vec1 * norm_vec2))
            # count_layer_conflict.append(torch.cosine_similarity(torch.from_numpy(temp_1), torch.from_numpy(temp_2))) # 计算余弦相似度

        return count_layer_conflict