# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                     #
#   BSD 2-Clause License                                                              #
#                                                                                     #
#   Copyright (c) 2021, Intelligent Systems Lab @ University of Oxford                #
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


import typing

import torch
import torch.nn as nn


__copyright__ = "Intelligent Systems Lab @ University of Oxford"
__license__ = "BSD-2-Clause"
__version__ = "0.2.0-alpha.2"
__date__ = "21 Mar 2021"
__status__ = "Development"


class PCLayer(nn.Module):
    """PCLayer.

        PCLayer should be inserted between layers where you want the error to be propagated
            in the predictive coding's (PC's) way, instead of the backpropagation's (BP's) way.
    """

    def __init__(
        self,
        energy_fn: typing.Callable = lambda vars: 0.5 * (vars['mu'] - vars['x'])**2,
        sample_x_fn: typing.Callable = lambda vars: vars['mu'],
    ):
        """Creates a new instance of ``PCLayer``.

        Args:
            energy_fn: The fn that specifies the how to compute the energy of error.
            sample_x_fn: The fn that specifies the how to sample x from mu.
        """

        super().__init__()

        assert callable(energy_fn)
        self._energy_fn = energy_fn

        assert callable(sample_x_fn)
        self._sample_x_fn = sample_x_fn

        # create all required parameters and buffers
        self._energy = []
        self._is_sample_x = False
        self._x = None

        # initially, we set the module in evaluation mode
        self.eval()

    #  GETTERS & SETTERS  ####################################################################################################

    def get_is_sample_x(self) -> bool:
        """
        """

        return self._is_sample_x

    def set_is_sample_x(self, is_sample_x: bool) -> None:
        """
        """

        assert isinstance(is_sample_x, bool)
        self._is_sample_x = is_sample_x

    def get_x(self) -> nn.Parameter:
        """
        """

        return self._x

    #  METHODS  ##############################################################################################################

    def reset_energy(self) -> None:
        """"""
        self._energy = []

    def energy(self) -> list:
        """The energy held by this PCLayer (summed over all other dimensions except batch dimension).
            It is a list with each element being one of the above energy since <reset_energy()>.
        """

        return [energy for energy in self._energy]

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            mu: The input.

        Returns:
            The output.
        """

        # sanitize args
        assert isinstance(mu, torch.Tensor)

        if self.training:

            # sample_x
            if self._is_sample_x:  # -> x has to be sampled

                x_data = self._sample_x_fn(
                    {
                        'mu': mu.detach(),
                        'x': None if self._x is None else self._x.data.detach(),
                    }
                )

                self._x = nn.Parameter(x_data.to(mu.device), True)

            else:

                if self._x is None:
                    raise RuntimeError(
                        "The <pc_layer._x> has not been initialized yet, run with <pc_layer.set_is_sample_x(True)> first."
                    )

                else:
                    # detect changing device
                    assert mu.device == self._x.device
                    # detect changing size
                    if mu.size() != self._x.size():
                        raise RuntimeError(
                            "You have changed the shape of this layer, you should do <pc_layer.set_is_sample_x(True) when changing the shape of this layer.\n"
                            "This should have been taken care of by <pc_trainer> unless you have set <is_sample_x_at_epoch_start=False> when calling <pc_trainer.train_on_batch()>, \n"
                            "when you should be responsible for making sure the batch size stays still."
                        )

            # energy, keep the batch dim, other dimensions are reduced to a single dimension
            energy = self._energy_fn(
                {
                    'x': self._x,
                    'mu': mu,
                }
            ).sum(
                dim=list(
                    range(len(mu.size()))
                )[1:],
                keepdim=False,
            ).unsqueeze(1)
            # [batch_size, 1]

            self._energy.append(energy)

            return self._x

        else:

            return mu

    #  PRIVATE METHODS  ######################################################################################################
