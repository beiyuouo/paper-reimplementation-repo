import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # (1, 28, 28) -> (16, 28, 28)
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # (16, 28, 28) -> (512, 28, 28)
        self.cov2 = nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, stride=1, padding=1)
        # (512, 28, 28) -> (64, 28, 28)
        self.cov3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*7*7, 5120)
        self.fc2 = nn.Linear(5120, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.cov1(x))
        x = F.relu(F.max_pool2d(self.cov2(x), 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(F.max_pool2d(self.cov3(x), 2))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.shape)
        x = x.view(-1, 64*7*7)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.softmax(self.fc4(x), dim=1)
        return y

