import os
import random

import numpy as np


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
    return DatasetSpiral(data[0]), DatasetSpiral(data[1])


class Dataset(object):
    def __init__(self):
        super(Dataset, self).__init__()

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DatasetSpiral(Dataset):
    def __init__(self, data: np.ndarray):
        super(DatasetSpiral, self).__init__()
        self.features = data[:, :2]
        self.labels = data[:, 2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


def default_collate_fn(batch):
    data, target = list(), list()
    for d, t in batch:
        data.append(d)
        target.append(t)
    return np.stack(data), np.stack(target)


class Dataloader(object):
    def __init__(self, dataset=None, shuffle=False, batch_size=1, collate_fn=None):
        super(Dataloader, self).__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.index = None
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = default_collate_fn

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        index = [i for i in range(len(self.dataset))]
        if self.shuffle:
            random.shuffle(index)
        new_index = list()
        for i in range(len(self.dataset)):
            nl = list()
            for j in range(self.batch_size):
                nl.append(index[i])
            new_index.append(nl)

        self.index = iter(new_index)
        return self

    def __next__(self):
        output = list()
        index = next(self.index)
        for _i in index:
            output.append(self.dataset[_i])
        return self.collate_fn(output)


# class DataloaderSpiral(Dataloader):
#     def __init__(self, dataset=None, shuffle=True, batch_size=1, collate_fn=None):
#         super(DataloaderSpiral, self).__init__()
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.index = None
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn

# def dataloader(dataset=None, shuffle=True, batch_size=1):
#     dataset = np.copy(dataset)
#     if shuffle:
#         np.random.shuffle(dataset)
#     dataset = dataset.reshape((-1, batch_size, dataset.shape[1]))
#     return iter(dataset)
#
#
# def data_label_split(_data):
#     return _data[:, :2], _data[:, 2]
