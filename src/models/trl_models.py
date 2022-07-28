'''
Created Date: Wednesday, July 27th 2022, 2:09:31 pm
Author: Rutuja Gurav (rutuja.gurav@email.ucr.edu)
Copyright (c) 2022 M.A.D. Lab @ UCR (https://madlab.cs.ucr.edu)

'''

import torch.nn as nn
import torch.nn.functional as F
from tltorch.factorized_layers import TRL

class TRL4CNN(nn.Module):
    def __init__(self, num_classes):
        super(TRL4CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.trl1 = TRL(input_shape=(64, 5, 5), output_shape=(32, 5, 5), rank='same')
        # self.trl2 = TRL(input_shape=(32, 5, 5), output_shape=(16, 5, 5), rank='same')
        # self.trl3 = TRL(input_shape=(16, 5, 5), output_shape=(8, 5, 5), rank='same')
        
        self.fc1 = nn.LazyLinear(128) # 128 is a random choice by me could be anything
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.max_pool1(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.max_pool2(out))

        out = self.trl1(out)
        # out = self.trl2(out)
                
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out