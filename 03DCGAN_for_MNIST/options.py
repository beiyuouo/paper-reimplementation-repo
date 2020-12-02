import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=10, help="epochs of training")
    args.add_argument('--bs', type=int, default=64, help='input batch size')
    args.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    args.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
    args.add_argument('--model', type=str, default='gan', help='model name')
    args.add_argument('--dataset', type=str, default='mnist', help='model name')
    args.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args.add_argument('--root', type=str, default='./data', help='path to dataset')
    args.add_argument('--log', type=str, default='./log', help='path to log')
    args.add_argument('--output', type=str, default='./result', help='path to output')
    args.add_argument('--train', type=str, default='yes', help='train or not')
    args.add_argument('--recover', type=str, default='none', help='recover or not')

    args.add_argument('--grid_w', type=int, default=16, help='result grid width')
    args.add_argument('--grid_h', type=int, default=4, help='result grid height')

    args.add_argument('--rvs', type=int, default=64, help='random vector size')
    args.add_argument('--insp', type=int, default=28, help='input feature')
    args.add_argument('--inch', type=int, default=1, help='D input channel')
    args.add_argument('--outf', type=int, default=128, help='D output feature')

    args = args.parse_args()
    return args
