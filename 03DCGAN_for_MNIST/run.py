import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

from model import GAN
from options import get_args
from data import get_data

if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    try:
        os.makedirs(args.output)
    except OSError:
        pass
    try:
        os.makedirs(args.log)
    except OSError:
        pass

    train_loader, test_loader = get_data(args)
    
    model = GAN(args=args)
    model.train(train_loader)
    # model.test(test_loader)



