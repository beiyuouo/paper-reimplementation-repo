import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

from model import CNN
from options import get_args
from data import get_data
from util import get_grid

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 28*28
    output_size = args.num_classes
    model = CNN(input_size=input_size, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        correct = 0
        total = 0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred_t = torch.max(y_pred.data, 1)
            total += y.size(0)
            # print(y_pred_t, y.data)
            # print((y_pred_t == y).sum().item())
            correct += (y_pred_t == y).sum().item()
            # print(correct, total)

            loss = criterion(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx + 1) % 100 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                      .format(epoch + 1, args.epochs, idx + 1, len(train_loader), loss.item(), 100 * correct / total))

    torch.save(model.state_dict(), os.path.join('./log', '{}_{}_{}.ckpt'.format(args.model, args.dataset, args.epochs)))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()
            # print(result)
            if idx % 100 == 0:
                get_grid(x.cpu().numpy(), args, args.epochs, idx)
                print(y_pred.data.cpu().numpy(), y.data.cpu().numpy)
        print('Test Acc: {:.4f}%, Model: {}, Epochs: {}'.format(correct/total*100, args.model, args.epochs))




