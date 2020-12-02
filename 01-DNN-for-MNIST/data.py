import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import pandas as pd
import numpy as np

import os


def get_data(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])),
        batch_size=args.bs, shuffle=True)
    return train_loader, test_loader

from options import get_args
from util import get_grid

if __name__ == '__main__':
    args = get_args()
    # print(args.bs)
    # print(args.root)
    train_loader, test_loader = get_data(args=args)
    for idx, (x, y) in enumerate(train_loader):
        # print(x)
        # print(x.shape)
        imgs = get_grid(x, args, 1, idx)
        # imgs.show()

