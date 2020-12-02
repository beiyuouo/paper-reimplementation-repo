import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 5120)
        self.fc2 = nn.Linear(5120, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(512, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        y = F.softmax(self.fc5(x), dim=1)
        return y

