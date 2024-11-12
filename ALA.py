import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple


class ALA:
    def __init__(self,
                 loss: nn.Module,
                 train_data, 
                 ALA_round,
                 batch_size: int, 
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu') -> None:
        """
        Initialize ALA module.
        """
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.layer_idx = layer_idx
        self.eta = eta
        self.ala_round = ALA_round
        self.device = device
        self.weights = None  # Learnable local aggregation weights
        self.start_phase = True

    def adaptive_local_aggregation(self, global_model: nn.Module, local_model: nn.Module) -> None:
        """
        Perform adaptive local aggregation using sampled local training data.
        """

        # Get parameters from models
        global_params = list(global_model.parameters())
        local_params = list(local_model.parameters())

        # Skip aggregation if parameters match
        if torch.sum(global_params[0] - local_params[0]) == 0:
            return

        # Preserve updates in lower layers
        for local_param, global_param in zip(local_params[:-self.layer_idx], global_params[:-self.layer_idx]):
            local_param.data = global_param.data.clone()

        # Prepare temporary model for weight learning
        temp_model = copy.deepcopy(local_model)
        temp_params = list(temp_model.parameters())
        higher_layer_params = local_params[-self.layer_idx:]
        global_higher_layer_params = global_params[-self.layer_idx:]
        temp_higher_layer_params = temp_params[-self.layer_idx:]

        # Freeze lower layers
        for param in temp_params[:-self.layer_idx]:
            param.requires_grad = False

        # Initialize weights if not already set
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in higher_layer_params]

        # Set up optimizer (no learning rate since we won't call step)
        optimizer = torch.optim.SGD(temp_higher_layer_params, lr=0)

        # Weight learning loop
        #losses = []  # Track losses

        for i in range(self.ala_round):
            #running_loss = 0.0
            for x, y in self.train_data:
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output,_ = temp_model(x)
                loss_value = self.loss(output, y)  # Calculate loss
                loss_value.backward()

                # Update weights
                for temp_param, local_param, global_param, weight in zip(temp_higher_layer_params, higher_layer_params, global_higher_layer_params, self.weights):
                    weight.data = torch.clamp(weight - self.eta * (temp_param.grad * (global_param - local_param)), 0, 1)

                # Update temporary model
                for temp_param, local_param, global_param, weight in zip(temp_higher_layer_params, higher_layer_params, global_higher_layer_params, self.weights):
                    temp_param.data = local_param + (global_param - local_param) * weight
                    
                #running_loss += loss_value.item()

            #losses.append(loss_value.item())
            #print(f'ALA_round: {i+1}, Loss: {running_loss/len(self.train_data)}')


        self.start_phase = False

        # Update local model with trained weights
        for local_param, temp_param in zip(higher_layer_params, temp_higher_layer_params):
            local_param.data = temp_param.data.clone()
