# -*- coding: utf-8 -*-
# @Time : 2022/4/23 19:59
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import torch
from torch import nn
from torch.nn import Module
from torch.autograd import Function


class MLPBlock(Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden_dims=None):
        super(MLPBlock, self).__init__()
        if hidden_dims is None:
            hidden_dims = [100, 100]

        layer_all = [in_dim] + hidden_dims + [out_dim]

        self.block = nn.Sequential()

        for i in range(len(layer_all) - 1):
            self.block.add_module(f"linear{i}",
                                  nn.Linear(in_features=layer_all[i], out_features=layer_all[i + 1])
                                  )
            self.block.add_module(f"bn{i}",
                                  nn.BatchNorm1d(layer_all[i + 1]))
            self.block.add_module(f"relu{i}",
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.block(x)


class FeatureExtractor(Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden_dims=None):
        super(FeatureExtractor, self).__init__()
        self.block = MLPBlock(in_dim, out_dim, hidden_dims)

    def forward(self, x):
        return self.block(x)


class LabelClassifier(Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden_dims=None):
        super(LabelClassifier, self).__init__()
        dim_hidden_last = hidden_dims.pop(-1)
        self.block = MLPBlock(in_dim, dim_hidden_last, hidden_dims)
        self.linear_out = nn.Linear(in_features=dim_hidden_last, out_features=out_dim)

    def forward(self, x):
        x = self.block(x)
        x = self.linear_out(x)
        return x


class DomainClassifier(Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden_dims=None):
        super(DomainClassifier, self).__init__()
        dim_hidden_last = hidden_dims.pop(-1)
        self.block = MLPBlock(in_dim, dim_hidden_last, hidden_dims)
        self.linear_out = nn.Linear(in_features=dim_hidden_last, out_features=out_dim)

    def forward(self, x):
        x = self.block(x)
        x = self.linear_out(x)
        return x


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float = 1.):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        alpha = ctx.alpha
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.neg() * alpha
        return grad_input, None


def grad_reverse(x, alpha: float = 1):
    return RevGrad.apply(x, alpha)


class DANN(Module):
    def __init__(self, in_dim: int = 1, label_dim: int = 1, domain_dim: int = 1, feature_dim: int = 128,
                 feature_extractor_dims=None, label_classifier_dims=None, domain_classifier_dims=None,
                 ):
        super(DANN, self).__init__()
        if feature_extractor_dims is None:
            feature_extractor_dims = [128]
        if label_classifier_dims is None:
            label_classifier_dims = [64, 64]
        if domain_classifier_dims is None:
            domain_classifier_dims = [64, 64]

        self.feature_extractor = FeatureExtractor(in_dim=in_dim, out_dim=feature_dim,
                                                  hidden_dims=feature_extractor_dims)
        self.label_classifier = LabelClassifier(in_dim=feature_dim, out_dim=label_dim,
                                                hidden_dims=label_classifier_dims)
        self.domain_classifier = DomainClassifier(in_dim=feature_dim, out_dim=domain_dim,
                                                  hidden_dims=domain_classifier_dims)

        self.source = True

    def set_source(self):
        self.source = True

    def set_target(self):
        self.source = False

    def forward(self, x, alpha: float = 0):
        feature = self.feature_extractor(x)
        if not self.training:
            output_label = self.label_classifier(feature)
            output_domain = self.domain_classifier(feature)
            return output_label, output_domain

        feature_reverse = grad_reverse(feature, alpha)
        output_domain = self.domain_classifier(feature_reverse)

        if self.source:
            output_label = self.label_classifier(feature)
            return output_label, output_domain
        else:
            return output_domain


