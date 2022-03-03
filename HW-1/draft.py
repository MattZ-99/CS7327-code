import os
import numpy as np


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


def load_data(data_dir="./data/"):
    data_file_list = ["two_spiral_train_data-1.txt", "two_spiral_test_data-1.txt"]
    data = list()
    for file_name in data_file_list:
        with open(os.path.join(data_dir, file_name), 'r') as file:
            content = file.readlines()
            content = [c.rstrip().split() for c in content]
            str2float = lambda cnt: [float(c) for c in cnt]
            content = [str2float(cnt) for cnt in content]
            data.append(np.array(content))
    return data[0], data[1]


if __name__ == "__main__":
    train_data, test_data = load_data()
    # print(train_data.shape, test_data.shape)
    print(train_data)
