import numpy as np


class Network(object):
    _training: bool = True

    def __init__(self):
        self.parameters_list = list()
        self.buffer = dict()
        super(Network, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def _make_parameters(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


class ModuleList(Network):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.module_list = list()

    def append(self, layer: Network):
        self.module_list.append(layer)

    # def __iter__(self):
    #     return iter(self.module_list)

    def __getitem__(self, item):
        return self.module_list[item]

    def __len__(self):
        return len(self.module_list)


class Parameters(object):
    data = None
    gradient = None

    def __init__(self, shape=None):
        super(Parameters, self).__init__()
        self.data = np.empty(shape)
        self.gradient = np.zeros(shape)

    def _gradient_zero(self):
        self.gradient.fill(0.)
