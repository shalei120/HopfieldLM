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


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoneModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoneModule, self).__init__()

    def forward(self, x):
        return x


class Softmax(nn.Softmax):
    def __init__(self):
        super(Softmax, self).__init__(dim=1)


class Softmin(nn.Softmin):
    def __init__(self):
        super(Softmin, self).__init__(dim=1)


class LogSoftmax(nn.LogSoftmax):
    def __init__(self):
        super(LogSoftmax, self).__init__(dim=1)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()


class SigmoidReLU(nn.Module):
    def forward(self, x):
        return torch.stack(
            [
                torch.sigmoid(x),
                (0.25 * x + 0.5)
            ],
            dim=0
        ).max(
            dim=0,
            keepdim=False,
        )[0]


class SigmoidReLU4(SigmoidReLU):
    def forward(self, x):
        return super(SigmoidReLU4, self).forward(4.0 * x)


class SigmoidReLU4P(SigmoidReLU4):
    def forward(self, x):
        return super(SigmoidReLU4P, self).forward(x - 0.5)


class X(nn.Module):
    """X.
        A module that holds x.
    """

    def __init__(self, x):
        """Initializer.

        Args:
            x: The x to hold.
        """
        super(X, self).__init__()

        self.x = x

    def forward(self, *input, **kwargs):
        """forward.

        The forward function will always return self.x.
        """
        return self.x
