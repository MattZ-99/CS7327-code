# -*- coding: utf-8 -*-
# @Time : 2022/4/29 13:56
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
from .DANN import FeatureExtractor, LabelClassifier


class MLP(Module):
    def __init__(self, in_dim: int = 1, label_dim: int = 1, feature_dim: int = 128,
                 feature_extractor_dims=None, label_classifier_dims=None,
                 ):
        super(MLP, self).__init__()
        if feature_extractor_dims is None:
            feature_extractor_dims = [128]
        if label_classifier_dims is None:
            label_classifier_dims = [64, 64]

        self.feature_extractor = FeatureExtractor(in_dim=in_dim, out_dim=feature_dim,
                                                  hidden_dims=feature_extractor_dims)
        self.label_classifier = LabelClassifier(in_dim=feature_dim, out_dim=label_dim,
                                                hidden_dims=label_classifier_dims)

    def forward(self, x):
        feature = self.feature_extractor(x)
        output_label = self.label_classifier(feature)

        return output_label
