import argparse
import time
import os


def get_timestamp(file_name=None):
    localtime = time.localtime(time.time())
    date_time = "{}_{}_{}_{}_{}_{}".format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                           localtime.tm_min, localtime.tm_sec)
    if file_name:
        return file_name + '_' + date_time
    return date_time


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_args():
    parser = argparse.ArgumentParser(description='CS7327-HW1')

    parser.add_argument("--seed", type=int, default=20220304, help="Seed for everything.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number.")
    parser.add_argument("--Epochs", type=int, default=1000, help="Total epochs.")
    parser.add_argument('-lr', type=float, default=0.01, help="Learning rate.")

    args = parser.parse_args()
    return args


def get_parser():
    parser = argparse.ArgumentParser(description='CS7327-HW1')

    parser.add_argument("--seed", type=int, default=20220304, help="Seed for everything.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number.")
    parser.add_argument("--Epochs", type=int, default=1000, help="Total epochs.")
    parser.add_argument('-lr', type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--min-max-data-split-mode", type=str, default="random", help="MinMax module split mode.")

    return parser


def seed_everything(seed: int = 0):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ValueStat:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, v=0, n=1):
        self.value += v * n
        self.count += n

    def reset(self):
        self.value = 0
        self.count = 0

    def get_sum(self):
        return self.value

    def get_avg(self):
        if self.count == 0:
            return -1
        return self.value / self.count

