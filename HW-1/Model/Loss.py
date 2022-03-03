import numpy as np

from .Module import Network


class _Loss(Network):
    def __init__(self, reduction="mean"):
        super(_Loss, self).__init__()
        self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: np.ndarray, target: np.ndarray):
        loss = (input - target) ** 2
        if self.reduction == "mean":
            self.buffer["gradient"] = 1. / loss.shape[0] * (input - target)
            return np.mean(loss)
        if self.reduction == "sum":
            self.buffer["gradient"] = input - target
            return np.sum(loss)
        else:
            raise NotImplementedError

    def backward(self):
        return self.buffer["gradient"]
