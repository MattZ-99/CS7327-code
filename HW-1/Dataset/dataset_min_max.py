import os
import random
from typing import List, Union, Tuple, Iterable
import math
from .dataset import Dataset, DatasetSpiral
import numpy as np


def load_data(data_dir: str = "./data/", trainset_split_shape: Union[tuple, None] = None,
              trainset_split_mode: str = "random"):
    data_file_list = ["two_spiral_train_data-1.txt", "two_spiral_test_data-1.txt"]
    data = list()
    for file_name in data_file_list:
        with open(os.path.join(data_dir, file_name), 'r') as file:
            content = file.readlines()
        content = [c.rstrip().split() for c in content]
        str2float = lambda cnt: [float(c) for c in cnt]
        content = [str2float(cnt) for cnt in content]
        data.append(np.array(content))
    return DatasetListSpiral(data[0], split_shape=trainset_split_shape, split_mode=trainset_split_mode), \
           DatasetSpiral(data[1])


class DatasetListSplit(object):
    elements: List[list] = None
    shape: Union[Tuple, None] = None

    def __init__(self):
        super(DatasetListSplit, self).__init__()

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DatasetListSpiral(DatasetListSplit):
    def __init__(self, data: np.ndarray = None, split_shape: Iterable = None, split_mode: str = "random"):
        super(DatasetListSpiral, self).__init__()

        if data is None:
            split_shape = None
            self.__len = 0
        else:
            self.__len = len(data)

        if split_shape is not None:
            self.shape = tuple(split_shape)
        if self.shape is None and data is not None:
            self.shape = tuple([1, 1])

        self._dataset_split(data, split_mode)

    def _dataset_split(self, data: np.ndarray = None, split_mode: str = "random"):
        index_positive = np.squeeze(np.argwhere(data[:, 2]))
        index_negative = np.squeeze(np.argwhere(1 - data[:, 2]))
        data_positive = data[index_positive]
        data_negative = data[index_negative]

        # print(data_positive.shape, data_negative.shape)
        if split_mode == "random":
            def random_split(data: np.ndarray, num: int) -> List[np.ndarray]:
                index = np.random.permutation(data.shape[0])
                element_len = len(index) // num
                splited_list = list()
                for i in range(num):
                    splited_list.append(data[index[element_len * i: element_len * (i + 1)]])
                return splited_list

            data_list_positive = random_split(data=data_positive, num=self.shape[0])
            data_list_negative = random_split(data=data_negative, num=self.shape[1])
        elif split_mode == "circle":
            def circle_split(data: np.ndarray, num: int) -> List[np.ndarray]:
                index = np.argsort(np.sum(data[:, :2] ** 2, axis=1) ** (1 / 2))
                element_len = len(index) // num
                splited_list = list()
                for i in range(num):
                    splited_list.append(data[index[element_len * i: element_len * (i + 1)]])
                return splited_list

            data_list_positive = circle_split(data=data_positive, num=self.shape[0])
            data_list_negative = circle_split(data=data_negative, num=self.shape[1])
        elif split_mode == "parallel":
            def parallel_split(data: np.ndarray, num: int) -> List[np.ndarray]:
                index = np.argsort(data[:, 0])
                element_len = len(index) // num
                splited_list = list()
                for i in range(num):
                    splited_list.append(data[index[element_len * i: element_len * (i + 1)]])
                return splited_list

            data_list_positive = parallel_split(data=data_positive, num=self.shape[0])
            data_list_negative = parallel_split(data=data_negative, num=self.shape[1])
        elif split_mode == "benz":
            def benz_split(data: np.ndarray, num: int) -> List[np.ndarray]:
                data_cos = data[:, 1] / (np.sum(data[:, :2] ** 2, axis=1) ** (1 / 2))
                data_angle = np.arcsin(data_cos)
                data_angle = np.where(data[:, 0] < 0, data_angle, 2 * math.pi - data_angle)
                index = np.argsort(data_angle)

                element_len = len(index) // num
                splited_list = list()
                for i in range(num):
                    splited_list.append(data[index[element_len * i: element_len * (i + 1)]])
                return splited_list
            data_list_positive = benz_split(data=data_positive, num=self.shape[0])
            data_list_negative = benz_split(data=data_negative, num=self.shape[1])

        else:
            raise NotImplementedError

        if self.elements is None:
            self.elements = list()
        self.elements.append(data_list_positive)
        self.elements.append(data_list_negative)

    def __getitem__(self, item: tuple):
        data_positive = self.elements[0][item[0]]
        data_negative = self.elements[1][item[1]]
        return DatasetSpiral(np.concatenate([data_positive, data_negative]))

    def __len__(self):
        return self.__len

    def get_whole_dataset(self):
        result = list()
        for element in self.elements:
            result.extend(element)
        return DatasetSpiral(np.concatenate(result))
