from Dataset.dataset_min_max import load_data
import numpy as np
import copy


class A(object):
    e1 = 1

    def __init__(self):
        print("Create A.")
        self.e2 = 1

    def __eq__(self, other):
        return A()


def draft(b: A):
    b.e1 = 2
    b.e2 = 3


if __name__ == "__main__":
    a = A()
    print(a.e1, a.e2)
    draft(a)
    print(a.e1, a.e2)
    b = a
    print(b.e1, b.e2)