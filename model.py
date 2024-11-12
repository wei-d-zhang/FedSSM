# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import matplotlib.image as plimg
from   PIL import Image 
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms,models
import glob,os
import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset

############################################################################## cifar-10 #########################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.fc(out)
        return out,features


def cifar10_ResNet18():
    return ResNet(ResidualBlock)


############################################################################ Fashion_Mnist ###################################################################################
    
class Fashion_ResNet18(nn.Module):
    def __init__(self):
        super(Fashion_ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        # Get features from the penultimate layer
        features = self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(self.resnet.conv1(x)))))
        features = nn.functional.avg_pool2d(features, features.size()[3]).view(features.size(0), -1)
        logits = self.resnet.fc(features)
        return logits, features
    
    
################################################################### cifar-100 ##########################################################################################
class cifar100_ResNet18(nn.Module):
    def __init__(self):
        super(cifar100_ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=100)

    def forward(self, x):
        # Get features from the penultimate layer
        features = self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(self.resnet.conv1(x)))))
        features = nn.functional.avg_pool2d(features, features.size()[3]).view(features.size(0), -1)
        logits = self.resnet.fc(features)
        return logits, features


