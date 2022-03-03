from .Module import Network, Parameters


class Optimizer(object):
    def __init__(self, network: Network, lr=0.001):
        super(Optimizer, self).__init__()
        self.network = network
        self.lr = lr

    def zero_grad(self):
        ModuleList_flag = 0
        for para in self.network.parameters_list:
            module = self.network
            for p in para.split('.'):
                if p == "__ModuleList__":
                    ModuleList_flag = 1
                    continue
                if ModuleList_flag:
                    module = module[int(p)]
                    ModuleList_flag = 0
                    continue
                module = getattr(module, p)
            if isinstance(module, Parameters):
                module._gradient_zero()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, network: Network, lr=0.001):
        super(SGD, self).__init__(network, lr)

    def step(self):
        ModuleList_flag = 0
        for para in self.network.parameters_list:
            module = self.network
            for p in para.split('.'):
                if p == "__ModuleList__":
                    ModuleList_flag = 1
                    continue
                if ModuleList_flag:
                    module = module[int(p)]
                    ModuleList_flag = 0
                    continue
                module = getattr(module, p)
            if isinstance(module, Parameters):
                module.data -= self.lr * module.gradient


if __name__ == "__main__":
    optimizer = Optimizer()
    optimizer.zero_grad()
