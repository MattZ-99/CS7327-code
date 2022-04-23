# -*- coding: utf-8 -*-
# @Time : 2022/4/23 14:20
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(path='data.pkl'):
    with open(path, 'rb') as f:
        _data = pickle.load(f)

    return _data


class DatasetSeed(Dataset):

    class_dict = {
        -1: 0,
        0: 1,
        1: 2
    }

    def __init__(self, data, label):
        super(DatasetSeed, self).__init__()
        self.data = data
        self.index = self.label_to_index(label)

    def __getitem__(self, item):
        return self.data[item], self.index[item]

    def __len__(self):
        return len(self.index)

    def get_data_dim(self):
        return self.data.shape[1]

    @staticmethod
    def get_label_dim():
        return len(DatasetSeed.class_dict)

    @staticmethod
    def label_to_index(label):
        index = torch.empty_like(label)
        for k in DatasetSeed.class_dict:
            index[label == k] = DatasetSeed.class_dict[k]
        return index




class DataGenerator:
    def __init__(self, path: str, *args_dataloader, **kwargs_dataloader):
        self.data = load_data(path=path)
        self.data_keys = list(self.data.keys())
        self.args_dataloader = args_dataloader
        self.kwargs_dataloader = kwargs_dataloader

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, item):
        train_keys = self.data_keys.copy()
        test_keys = [train_keys.pop(item)]
        _train_dataloader = self._generate_dataloader(train_keys)
        _test_dataloader = self._generate_dataloader(test_keys)
        return _train_dataloader, _test_dataloader

    def _generate_data_array(self, index_keys: list):
        res_data_list = list()
        res_label_list = list()
        for index in index_keys:
            res_data_list.append(self.data[index]['data'])
            res_label_list.append(self.data[index]['label'])
        return np.concatenate(res_data_list), np.concatenate(res_label_list)

    def _generate_dataset(self, index_keys: list):
        data_array, label_array = self._generate_data_array(index_keys)
        data_array = torch.from_numpy(data_array).float()
        label_array = torch.from_numpy(label_array).long()

        dataset = DatasetSeed(data_array, label_array)

        return dataset

    def _generate_dataloader(self, index_keys: list):
        dataset = self._generate_dataset(index_keys)
        dataloader = DataLoader(dataset, *self.args_dataloader, **self.kwargs_dataloader)
        return dataloader

    def gen_dataloader(self, item, source: bool = True, *args_dataloader, **kwargs_dataloader):
        train_keys = self.data_keys.copy()
        test_keys = [train_keys.pop(item)]
        if source:
            return self._generate_dataloader_with_args(train_keys, *args_dataloader, **kwargs_dataloader)
        else:
            return self._generate_dataloader_with_args(test_keys, *args_dataloader, **kwargs_dataloader)

    def _generate_dataloader_with_args(self, index_keys: list, *args_dataloader, **kwargs_dataloader):
        dataset = self._generate_dataset(index_keys)
        dataloader = DataLoader(dataset, *args_dataloader, **kwargs_dataloader)
        return dataloader


if __name__ == '__main__':
    dataGen = DataGenerator('../data/data.pkl', batch_size=4, drop_last=False)
    train_dataloader, test_dataloader = dataGen[0]

    # for data, label in train_dataloader:
    #     print(data.shape, label.shape)
