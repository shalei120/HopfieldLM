# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                     #
#   BSD 2-Clause License                                                              #
#                                                                                     #
#   Copyright (c) 2020, Intelligent Systems Lab @ University of Oxford                #
#   All rights reserved.                                                              #
#                                                                                     #
#   Redistribution and use in source and binary forms, with or without                #
#   modification, are permitted provided that the following conditions are met:       #
#                                                                                     #
#   1. Redistributions of source code must retain the above copyright notice, this    #
#      list of conditions and the following disclaimer.                               #
#                                                                                     #
#   2. Redistributions in binary form must reproduce the above copyright notice,      #
#      this list of conditions and the following disclaimer in the documentation      #
#      and/or other materials provided with the distribution.                         #
#                                                                                     #
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"       #
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE         #
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    #
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE      #
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        #
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR        #
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,     #
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE     #
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import abc
import math

import numpy as np
import torch
import torch.nn as nn


class SparseNeurons(nn.Module):
    """
    SparseNeurons
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, size, neuronSparsity, normalize_sparse_neuron):
        """
        Enforce neuron sparsity during training.

        Sample usage:

          model = nn.Sequential([
            nn.Linear(784, 10)
            SparseNeuron([10],0.4)
          ])

        :param size:
          The size to sparsify the neurons.
        :param neuronSparsity:
          Pct of neurons that are allowed to be non-zero in the layer.
        """
        super(SparseNeurons, self).__init__()

        assert (
            size.__class__ in [list, tuple]
        ), 'The size needs to be a list or tuple.'

        self.size = list(size)
        self.out_size = self.size

        assert isinstance(
            neuronSparsity, float
        ), 'The neuronSparsity needs to be a float.'

        assert (
            0 < neuronSparsity < 1
        ), 'The neuronSparsity needs to be between 0 (exclusive) and 1 (exclusive).'

        self.neuronSparsity = neuronSparsity

        self.binomial = torch.distributions.binomial.Binomial(
            probs=self.neuronSparsity
        )

        self.resetIndices()

        assert isinstance(
            normalize_sparse_neuron, bool
        ), 'The normalize_sparse_neuron needs to be a bool.'

        self.normalize_sparse_neuron = normalize_sparse_neuron

    def forward(self, x):
        if self.to_resetIndices:
            self.zeroNrs = self.computeIndices(x)
            self.to_resetIndices = False

        if self.training:
            x = self.rezeroNeuron(x)
            if self.normalize_sparse_neuron:
                x = x * (1.0 / self.neuronSparsity)

        return x

    def computeIndices(self, x):
        """
        For each unit, decide which neurons are going to be zero
        :return: tensor mask for all non-zero neurons. See :meth:`rezeroNeuron`
        """
        zeroNrs = self.binomial.sample(
            x.size()
        ).to(x.device)
        return zeroNrs

    def resetIndices(self):
        self.to_resetIndices = True

    def rezeroNeuron(self, x):
        """
        Set the previously selected neurons to zero. See :meth:`computeIndices` and :meth:`resetIndices`
        """
        return x * self.zeroNrs
