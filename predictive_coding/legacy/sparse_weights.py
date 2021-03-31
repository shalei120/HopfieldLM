# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


import abc
import math

import numpy as np
import torch
import torch.nn as nn


# def rezeroWeights(m):
#     """
#     Function used to update the weights after each epoch.
#
#     Call using :meth:`torch.nn.Module.apply` after each epoch if required
#     For example: ``m.apply(rezeroWeights)``
#
#     :param m: SparseWeightsBase module
#     """
#     if isinstance(m, SparseWeightsBase):
#         if m.training:
#             m.rezeroWeights()


# def normalizeSparseWeights(m):
#     """
#     Initialize the weights using kaiming_uniform initialization normalized to
#     the number of non-zeros in the layer instead of the whole input size.
#     Similar to torch.nn.Linear.reset_parameters() but applying weight sparsity
#     to the input size
#     """
#     if isinstance(m, SparseWeightsBase):
#         _, inputSize = m.module.weight.shape
#         fan = int(inputSize * m.weightSparsity)
#         gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
#         std = gain / np.math.sqrt(fan)
#         # Calculate uniform bounds from standard deviation
#         bound = math.sqrt(3.0) * std
#         nn.init.uniform_(m.module.weight, -bound, bound)
#         if m.module.bias is not None:
#             bound = 1 / math.sqrt(fan)
#             nn.init.uniform_(m.module.bias, -bound, bound)


class SparseWeightsBase(nn.Module):
    """
    Base class for the all Sparse Weights modules
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, module, weightSparsity):
        """
        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeightsBase, self).__init__()

        assert (
            0 < weightSparsity < 1
        ), 'The weightSparsity needs to be between 0 (exclusive) and 1 (exclusive).'

        self.module = module
        self.weightSparsity = weightSparsity
        self.register_buffer("zeroWts", self.computeIndices())

        self.register_buffer("weight_data", self.module.weight.data.clone())

        # create references for easy access to the wrapped model
        if isinstance(self.module, nn.Linear):
            self.in_features = self.module.in_features
            self.out_features = self.module.out_features
        elif isinstance(self.module, nn.Conv2d):
            self.in_channels = self.module.in_channels
            self.out_channels = self.module.out_channels
        else:
            raise NotImplementedError
        if hasattr(self.module, 'out_size'):
            self.out_size = self.module.out_size
        self.weight = self.module.weight
        self.bias = self.module.bias

    def train(self, mode=True):
        """Switch to train or eval mode.

        If switching to train mode: zeroWts will be resampled, before which the weight will be recorded.
        """
        self = super(SparseWeightsBase, self).train(mode=mode)

        if self.training:

            # resetIndices

            # before resetIndices, we need to record self.module.weight
            # into self.weight_data
            self.rezeroWeights()
            mask = self.module.weight.data.sign().abs()
            self.weight_data = (
                self.weight_data * (1 - mask)
            ) + (
                self.module.weight.data
            )

            # then resetIndices
            self.resetIndices()

        # in each switching of mode, self.module.weight is set
        # back to the self.weight_data
        self.module.weight.data.copy_(self.weight_data)

        return self

    def forward(self, x):
        if self.training:
            self.rezeroWeights()
        return self.module.forward(x)

    @abc.abstractmethod
    def computeIndices(self):
        """
        For each unit, decide which weights are going to be zero
        :return: tensor indices for all non-zero weights. See :meth:`rezeroWeights`
        """
        raise NotImplementedError

    def resetIndices(self):
        self.zeroWts = self.computeIndices()

    @abc.abstractmethod
    def rezeroWeights(self):
        """
        Set the previously selected weights to zero. See :meth:`computeIndices`
        """
        raise NotImplementedError


class SparseWeights(SparseWeightsBase):
    def __init__(self, module, weightSparsity):
        """
        Enforce weight sparsity on linear module during training.

        Sample usage:

          model = nn.Linear(784, 10)
          model = SparseWeights(model, 0.4)

        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeights, self).__init__(
            module, weightSparsity)

    def computeIndices(self):
        # For each unit, decide which weights are going to be zero
        outputSize, inputSize = self.module.weight.shape
        numZeros = int(round((1.0 - self.weightSparsity) * inputSize))

        outputIndices = np.arange(outputSize)
        inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                                 for _ in outputIndices], dtype=np.long)

        # Create tensor indices for all non-zero weights
        zeroIndices = np.empty((outputSize, numZeros, 2), dtype=np.long)
        zeroIndices[:, :, 0] = outputIndices[:, None]
        zeroIndices[:, :, 1] = inputIndices
        zeroIndices = zeroIndices.reshape(-1, 2)
        return torch.from_numpy(zeroIndices.transpose())

    def rezeroWeights(self):
        zeroIdx = (self.zeroWts[0], self.zeroWts[1])
        self.module.weight.data[zeroIdx] = 0.0


class SparseWeights2d(SparseWeightsBase):
    def __init__(self, module, weightSparsity):
        """
        Enforce weight sparsity on CNN modules
        Sample usage:

          model = nn.Conv2d(in_channels, out_channels, kernel_size, ...)
          model = SparseWeights2d(model, 0.4)

        :param module:
          The module to sparsify the weights
        :param weightSparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(SparseWeights2d, self).__init__(
            module, weightSparsity)

    def computeIndices(self):
        # For each unit, decide which weights are going to be zero
        inChannels = self.module.in_channels
        outChannels = self.module.out_channels
        kernelSize = self.module.kernel_size

        inputSize = inChannels * kernelSize[0] * kernelSize[1]
        numZeros = int(round((1.0 - self.weightSparsity) * inputSize))

        outputIndices = np.arange(outChannels)
        inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                                 for _ in outputIndices], dtype=np.long)

        # Create tensor indices for all non-zero weights
        zeroIndices = np.empty((outChannels, numZeros, 2), dtype=np.long)
        zeroIndices[:, :, 0] = outputIndices[:, None]
        zeroIndices[:, :, 1] = inputIndices
        zeroIndices = zeroIndices.reshape(-1, 2)

        return torch.from_numpy(zeroIndices.transpose())

    def rezeroWeights(self):
        zeroIdx = (self.zeroWts[0], self.zeroWts[1])
        self.module.weight.data.view(
            self.module.out_channels, -1)[zeroIdx] = 0.0
