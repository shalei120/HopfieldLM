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
import warnings
import tqdm
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from . import pc_layer

__copyright__ = "Intelligent Systems Lab @ University of Oxford"
__license__ = "BSD-2-Clause"
__version__ = "0.2.0-alpha.2"
__date__ = "21 Mar 2021"
__status__ = "Development"


class PCTrainer(object):
    """A trainer for predictive-coding models that are implemented by means of
    :class:`pc_layer.PCLayer`s.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_x_fn: typing.Callable = optim.SGD,
        optimizer_x_kwargs: dict = {"lr": 0.1},
        manual_optimizer_x_fn: typing.Callable = None,
        x_lr_discount: float = 0.5,
        loss_x_fn: typing.Callable = None,
        loss_inputs_fn: typing.Callable = None,
        optimizer_p_fn: typing.Callable = optim.Adam,
        optimizer_p_kwargs: dict = {"lr": 0.001},
        manual_optimizer_p_fn: typing.Callable = None,
        T: int = 512,
        update_x_at: typing.Union[str, typing.List[int]] = "all",
        update_p_at: typing.Union[str, typing.List[int]] = "all",
        energy_coefficient: float = 1.0,
        early_stop_condition: str = "False",
        update_p_at_early_stop: bool = True,
        plot_progress_at: typing.Union[str, typing.List[int]] = "all",
    ):
        """Creates a new instance of ``PCTrainer``.

        Remind of notations:

            ------- h=0 ---------, ------- h=1 ---------
            t=0, t=1, ......, t=T, t=0, t=1, ......, t=T

            h: epoch. In each h, the same batch of data is presented, i.e., data batch is changed when h is changed.
            t: iteration. The integration step of inference.

        Args:
            model: The predictive-coding model to train.

            optimizer_x_fn: Callable to create optimizer of x.
            optimizer_x_kwargs: Keyword arguments for optimizer_x_fn.
            manual_optimizer_x_fn: Manually create optimizer_x.
                This will override optimizer_x_fn and optimizer_x_kwargs.

                See:
                ```python
                input('Start from zil and then il?')
                ```
                in demo.py as an example.

            x_lr_discount: Discount of learning rate of x if the overall does not decrease.
                Set to 1.0 to disable it.
                The goal of inference is to get things to convergence at the current
                batch of datapoints, which is different from the goal of updating parameters,
                which is to take a small step at the current batch of datapoints, so annealing
                the learning rate of x according to the overall is generally benefiting.
            loss_x_fn: Use this function to compute a loss from xs.
                This can be used, for example, for applying sparsity penalty to x:
                    <loss_x_fn=lambda x: 0.001 * x.abs().sum()>
            loss_inputs_fn: Use this function to compute a loss from inputs.
                Only takes effect when <is_optimize_inputs=True> when calling <self.train_on_batch()>.
                This can be used, for example, for applying sparsity penalty (pooled inhibit in the following example) to x:
                    <loss_inputs_fn=F.relu(x.abs().sum(1)-1).sum(0)>

            optimizer_p_fn: See optimizer_x_fn.
            optimizer_p_kwargs: See optimizer_x_kwargs.
            manual_optimizer_p_fn: See manual_optimizer_x_fn.

            T: Train on each sample for T times.
            update_x_at:
                If "all", update x during all t=0 to T-1.
                If "last", update x at t=T-1.
                If "last_half", update x during all t=T/2 to T-1.
                If list of int, update x at t in update_x_at.
            update_p_at: See update_x_at.

            energy_coefficient: The coefficient added to the energy.

            early_stop_condition: Early stop condition for <train_on_batch()>. It is a str and will be eval during and expected to produce a bool at the time.

            update_p_at_early_stop: When early stop is triggered, whether to update p at the iteration.

            plot_progress_at: Plot the progress of training at epochs.
                It could be a list of epochs (int) at which you want to plot the progress.
                It could be "all", which means to plot progress for all epochs.
                Note that the program will run in nonblocking mode, so that the figure is being recreated constantly.
                    To pause somewhere and investigate the figure closer, it is your responsibility to block your program somewhere, say, insert <input('pause')> at after a call of <train_on_batch()>.
        """

        assert isinstance(model, nn.Module)
        self._model = model

        assert callable(optimizer_x_fn)
        self._optimizer_x_fn = optimizer_x_fn

        assert isinstance(optimizer_x_kwargs, dict)
        self._optimizer_x_kwargs = optimizer_x_kwargs

        if manual_optimizer_x_fn is not None:
            assert callable(manual_optimizer_x_fn)
        self._manual_optimizer_x_fn = manual_optimizer_x_fn

        self._optimizer_x = None

        assert isinstance(x_lr_discount, float)
        assert x_lr_discount <= 1.0
        self._x_lr_discount = x_lr_discount

        if loss_x_fn is not None:
            assert callable(loss_x_fn)
        self._loss_x_fn = loss_x_fn
        if self._loss_x_fn is not None:
            assert self.get_is_model_has_pc_layers(), (
                "<loss_x_fn> should only work with models with <PCLayer>. "
            )

        if loss_inputs_fn is not None:
            assert callable(loss_inputs_fn)
        self._loss_inputs_fn = loss_inputs_fn
        if self._loss_inputs_fn is not None:
            assert self.get_is_model_has_pc_layers(), (
                "<loss_inputs_fn> should only work with models with <PCLayer>. "
            )

        assert callable(optimizer_p_fn)
        self._optimizer_p_fn = optimizer_p_fn

        assert isinstance(optimizer_p_kwargs, dict)
        self._optimizer_p_kwargs = optimizer_p_kwargs

        if manual_optimizer_p_fn is not None:
            assert callable(manual_optimizer_p_fn)
        self._manual_optimizer_p_fn = manual_optimizer_p_fn

        if self._manual_optimizer_p_fn is None:
            self._optimizer_p = self._optimizer_p_fn(
                self.get_model_parameters(),
                **self._optimizer_p_kwargs
            )
        else:
            self._optimizer_p = self._manual_optimizer_p_fn()

        assert isinstance(T, int)
        assert T > 0
        self._T = T

        if self.get_is_model_has_pc_layers():

            # ensure that T is compatible with the trained model
            if self._T < self.get_num_pc_layers() + 1:
                warnings.warn(
                    (
                        "You should always choose T such that T >= (<pc_trainer.get_num_pc_layers()> + 1), "
                        "as it ensures that the error can be PC-propagated through the network."
                    ),
                    category=RuntimeWarning
                )

            min_t = self.get_recommended_least_T()
            if self._T < min_t:
                warnings.warn(
                    (
                        "T is too small - just enough to PC-propagate the error through the network. "
                        "We recommend to use a minimum T of {min_t}"
                    ),
                    category=RuntimeWarning
                )

        update_x_at = self._preprocess_step_index_list(
            indices=update_x_at,
            T=self._T,
        )
        self._update_x_at = update_x_at

        update_p_at = self._preprocess_step_index_list(
            indices=update_p_at,
            T=self._T,
        )
        self._update_p_at = update_p_at

        assert isinstance(energy_coefficient, float)
        self._energy_coefficient = energy_coefficient

        assert isinstance(early_stop_condition, str)
        self._early_stop_condition = early_stop_condition

        assert isinstance(update_p_at_early_stop, bool)
        self._update_p_at_early_stop = update_p_at_early_stop

        if isinstance(plot_progress_at, str):
            assert plot_progress_at in ["all"]
        elif isinstance(plot_progress_at, list):
            for h in plot_progress_at:
                assert isinstance(h, int)
        else:
            raise NotImplementedError
        self._plot_progress_at = plot_progress_at

        self.reset_plot_progress()

    #  GETTERS & SETTERS  #####################################################################################################

    def get_model(self) -> nn.Module:
        return self._model

    def get_optimizer_x(self) -> optim.Optimizer:
        return self._optimizer_x

    def set_optimizer_x(self, optimizer_x: optim.Optimizer) -> None:
        assert isinstance(optimizer_x, optim.Optimizer)
        self._optimizer_x = optimizer_x

    def get_optimizer_p(self) -> optim.Optimizer:
        return self._optimizer_p

    def set_optimizer_p(self, optimizer_p: optim.Optimizer) -> None:
        assert isinstance(optimizer_p, optim.Optimizer)
        self._optimizer_p = optimizer_p

    def get_is_model_has_pc_layers(self) -> bool:
        """Evaluates if the trained model contains :class:`pc_layer.PCLayer."""

        for _ in self.get_model_pc_layers():
            return True
        else:
            return False

    def get_model_pc_layers_training(self) -> list:
        """Get a list of <pc_layer.training>."""

        pc_layers_training = []
        for pc_layer in self.get_model_pc_layers():
            pc_layers_training.append(pc_layer.training)
        return pc_layers_training

    def get_is_model_training(self):
        """Get whether the model is in train mode. This indicates that the model is in train mode
        and also all child pc_layers are in train mode. The same applies for eval mode.

        Returns:
            (bool | None):
                (bool): Whether the model is in train mode (True) or eval mode (False).
                (None): The model is neither in train mode nor eval mode,
                    because the child pc_layers are not in a unified state.
                    Calling <model.train()> or <model.eval()> is needed to unify the children pc_layers' states.
        """

        if (self._model.training) and np.all(self.get_model_pc_layers_training()):
            return True
        elif (not self._model.training) and np.all([not training for training in self.get_model_pc_layers_training()]):
            return False
        else:
            return None

    def get_energy(self) -> list:
        """Retrieves the energy held by each pc_layer."""

        energy = []
        for pc_layer in self.get_model_pc_layers():
            energy += pc_layer.energy()

        return energy

    def get_model_parameters(self) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieves the actual trainable parameters, which are all parameters except xs.
        """

        # fetch the xs
        all_model_xs = set(self.get_model_xs())

        # iterate over all parameters in the trained model, and retrieve those that are actually trained (i.e., exclude xs)
        for param in self._model.parameters():
            if not any(param is x for x in all_model_xs):
                yield param

    def get_model_pc_layers(self) -> typing.Generator[pc_layer.PCLayer, None, None]:
        """Retrieves all :class:`pc_layer.PCLayer`s contained in the trained model."""

        for module in self._model.modules():
            if isinstance(module, pc_layer.PCLayer):
                yield module

    def get_model_xs(self) -> typing.Generator[nn.Parameter, None, None]:
        """Retrieves xs.
        """

        for pc_layer in self.get_model_pc_layers():
            yield pc_layer.get_x()

    def get_num_pc_layers(self) -> int:
        """Computes the total number of :class:`pc_layer.PCLayer contained by the trained model."""

        return sum(1 for _ in self.get_model_pc_layers())

    def get_recommended_least_T(self) -> int:
        """Computes the minimum T recommended based on the number of :class:`pc_layer.PCLayer in the trained model."""

        return 4 * (self.get_num_pc_layers() + 1)

    #  METHODS  ########################################################################################################

    def reset_energy(self) -> None:
        """Reset the energy held by each pc_layer."""

        for pc_layer in self.get_model_pc_layers():
            pc_layer.reset_energy()

    def recreate_optimize_x(self) -> None:
        """Recreates the optimizer_x"""

        if self._manual_optimizer_x_fn is None:
            self._optimizer_x = self._optimizer_x_fn(
                self.get_model_xs(),
                **self._optimizer_x_kwargs
            )

        else:
            self._optimizer_x = self._manual_optimizer_x_fn()

    def reset_plot_progress(self):
        """Reset plot progress."""

        plt.ion()
        self._h = 0
        self._plot_progress = {
            "key": [],
            "h": [],
            "t": [],
            "value": [],
        }

    def train_on_batch(
        self,
        inputs: typing.Any,
        loss_fn: typing.Callable = None,
        loss_fn_kwargs: dict = {},
        is_sample_x_at_epoch_start: bool = True,
        is_reset_optimizer_x_at_epoch_start: bool = True,
        is_unwrap_inputs: bool = False,
        is_optimize_inputs: bool = False,
        callback_after_t: typing.Callable = None,
        callback_after_t_kwargs: dict = {},
        is_log_progress: bool = True,
        debug: dict = {},
    ):
        """Train on a batch.

        Args:

            inputs: This will be passed to self.model().

            loss_fn: The function that takes in
                    - the output of self.model
                    - the loss_fn_kwargs as keyword arguments
                and returns a loss.

            loss_fn_kwargs: The keyword arguments passed to loss_fn.

            is_sample_x_at_epoch_start: Whether to reset optimizer_x at the start of the epoch.
            is_reset_optimizer_x_at_epoch_start: Whether to reset optimizer_x at the start of the epoch.
                The default values of the above two arguments are True as we thought for each epoch from t=0 to t=T the inference is independent.
                    Specifically, we know that the batch of datapoints is fixed during t=0 to t=T, and will switch to another batch of datapoints from t=T:

                        ------- h=0 ---------, ------- h=1 ---------
                        t=0, t=1, ......, t=T, t=0, t=1, ......, t=T
                         |                    ^ |
                         |        switch the batch of datapoints
                         ^                      ^
                      at_epoch_start          at_epoch_start

                    Full batch training:
                        You may set these two arguments to False after the first epoch (h=0).

            is_unwrap_inputs: If unwrap inputs to be multiple arguments.

            is_optimize_inputs: If optimize inputs.

            callback_after_t: Callback functon after at the end of t. The function will taks in
                - t
                - callback_after_t_kwargs as keyword arguments

            callback_after_t_kwargs: This will be passed to callback_after_t() as keyword arguments.

            is_log_progress: If log progress of training.

            debug: For passing additional debug arguments.

        Returns:

            A dictionary containing:
                - lists: corresponds to progress during inference, with dimension variable being t
                - single values: corresponds to a single result
        """

        self.inputs = inputs

        # sanitize model
        assert (self.get_is_model_training() == True), (
            "PCLayer behaves differently in train and eval modes, like Dropout or Batch Normalization. "
            "Thus, call model.eval() before evaluation and model.train() before train. "
            "Make sure your model is in train mode before calling <train_on_batch()>. "
            "It can be done by calling <model.train()>. "
            "Do remember switching your model back to eval mode before evaluating it by calling <model.eval()>. "
        )

        # sanitize args
        if loss_fn is not None:
            assert callable(loss_fn)

        assert isinstance(loss_fn_kwargs, dict)

        assert isinstance(is_sample_x_at_epoch_start, bool)
        assert isinstance(is_reset_optimizer_x_at_epoch_start, bool)

        assert isinstance(is_unwrap_inputs, bool)
        if is_unwrap_inputs:
            assert isinstance(inputs, (tuple, list, dict))

        assert isinstance(is_optimize_inputs, bool)
        if is_optimize_inputs:
            assert self.get_is_model_has_pc_layers(), (
                "<is_optimize_inputs> should only work with models with <PCLayer>. "
            )
            assert (not is_unwrap_inputs)

        if callback_after_t is not None:
            assert callable(callback_after_t)

        assert isinstance(callback_after_t_kwargs, dict)

        assert isinstance(is_log_progress, bool)

        assert isinstance(debug, dict)

        # initialize the dict for storing results
        results = {
            "loss": [],
            "energy": [],
            "overall": [],
        }

        # create t_iterator
        if is_log_progress:
            t_iterator = tqdm.trange(self._T)
        else:
            t_iterator = range(self._T)

        for t in t_iterator:

            # -> inference

            # at_epoch_start
            if t == 0:

                if self.get_is_model_has_pc_layers():

                    # sample_x
                    if is_sample_x_at_epoch_start:
                        for pc_layer in self.get_model_pc_layers():
                            pc_layer.set_is_sample_x(True)

                    # optimize_inputs
                    if is_optimize_inputs:
                        # convert inputs to nn.Parameter
                        self.inputs = torch.nn.Parameter(self.inputs, True)

            # at_inference_start
            if self.get_is_model_has_pc_layers():
                # reset_energy
                self.reset_energy()

            # forward
            if is_unwrap_inputs:
                if isinstance(self.inputs, dict):
                    outputs = self._model(**self.inputs)
                elif isinstance(self.inputs, (list, tuple)):
                    outputs = self._model(*self.inputs)
                else:
                    raise NotImplementedError
            else:
                outputs = self._model(self.inputs)

            # at_epoch_start
            if t == 0:

                if self.get_is_model_has_pc_layers():

                    # sample_x
                    if is_sample_x_at_epoch_start:
                        # sample_x only takes effect for one inference step
                        for pc_layer in self.get_model_pc_layers():
                            pc_layer.set_is_sample_x(False)
                        # after sample_x, optimizer_x will be recreated
                        self.recreate_optimize_x()

                    # reset_optimizer_x
                    if is_reset_optimizer_x_at_epoch_start:
                        self.recreate_optimize_x()

                    # optimize_inputs
                    if is_optimize_inputs:
                        assert len(self._optimizer_x.param_groups) == 1
                        self._optimizer_x.param_groups[0]["params"].append(
                            self.inputs
                        )

            # loss
            if loss_fn is not None:
                loss = loss_fn(outputs, **loss_fn_kwargs)
                results["loss"].append(loss.item())
            else:
                loss = None

            # energy
            if self.get_is_model_has_pc_layers():
                energy = self.get_energy()
                energy_layer_batch_size_1 = torch.stack(
                    energy
                )
                # reduce the dimension of layer
                energy_batch_size_1 = energy_layer_batch_size_1.sum(
                    dim=0,
                    keepdim=False,
                )
                # reduce the dimensions of batch_size and 1
                energy = energy_batch_size_1.sum(
                    dim=[0, 1],
                    keepdim=False,
                )
                results["energy"].append(
                    energy.item()
                )
            else:
                energy = None

            # loss_x
            if self._loss_x_fn is not None:
                loss_x_layer = []
                for model_x in self.get_model_xs():
                    loss_x_layer.append(self._loss_x_fn(model_x))
                if len(loss_x_layer) > 0:
                    loss_x = torch.stack(loss_x_layer).sum()
                else:
                    loss_x = None
            else:
                loss_x = None

            # loss_inputs
            if self._loss_inputs_fn is not None:
                if is_optimize_inputs:
                    loss_inputs = self._loss_inputs_fn(self.inputs)
                else:
                    loss_inputs = None
            else:
                loss_inputs = None

            # overall
            overall = []
            if loss is not None:
                overall.append(loss)
            if energy is not None:
                overall.append(
                    energy * self._energy_coefficient
                )
            if loss_x is not None:
                overall.append(loss_x)
            if loss_inputs is not None:
                overall.append(loss_inputs)
            overall = sum(overall)
            results["overall"].append(overall.item())

            # zero_grad
            if self.get_is_model_has_pc_layers():
                if t in self._update_x_at:
                    self._optimizer_x.zero_grad()
            self._optimizer_p.zero_grad()

            # backward
            overall.backward()

            # early_stop
            early_stop = eval(self._early_stop_condition)

            # optimizer_x
            # x_lr_discount
            if self.get_is_model_has_pc_layers():
                if t in self._update_x_at:

                    # optimizer_x
                    self._optimizer_x.step()

                    # x_lr_discount
                    if self._x_lr_discount < 1.0:
                        if len(results["overall"]) >= 2:
                            if not (results["overall"][-1] < results["overall"][-2]):
                                for param_group_i in range(len(self._optimizer_x.param_groups)):
                                    self._optimizer_x.param_groups[param_group_i][
                                        'lr'
                                    ] = self._optimizer_x.param_groups[param_group_i][
                                        'lr'
                                    ] * self._x_lr_discount

            # optimizer_p
            if (t in self._update_p_at) or (early_stop and self._update_p_at_early_stop):
                self._optimizer_p.step()

            # callback_after_t
            if callback_after_t is not None:
                callback_after_t(t, **callback_after_t_kwargs)
                if not (self.get_is_model_training() == True):
                    raise RuntimeError(
                        "If you do <model.eval()> in <callback_after_t()>, you need to put model back to train mode when leaving <callback_after_t()>. "
                    )

            # log_progress
            if is_log_progress:
                log_progress = '|'
                if loss is not None:
                    log_progress += " l: {:.3e} |".format(
                        loss,
                    )
                if energy is not None:
                    log_progress += " e: {:.3e} |".format(
                        energy,
                    )
                if loss_x is not None:
                    log_progress += " x: {:.3e} |".format(
                        loss_x,
                    )
                if loss_inputs is not None:
                    log_progress += " i: {:.3e} |".format(
                        loss_inputs,
                    )
                log_progress += " o: {:.3e} |".format(
                    overall,
                )
                if self.get_is_model_has_pc_layers():
                    if self._x_lr_discount < 1.0:
                        x_lrs = []
                        for param_group_i in range(len(self._optimizer_x.param_groups)):
                            x_lrs.append(
                                self._optimizer_x.param_groups[param_group_i][
                                    'lr'
                                ]
                            )
                        log_progress += " x_lrs: {} |".format(
                            x_lrs,
                        )
                t_iterator.set_description(log_progress)

            # plot_progress
            if (isinstance(self._plot_progress_at, str) and self._plot_progress_at == "all") or (self._h in self._plot_progress_at):
                for key, result in results.items():
                    if isinstance(result, list) and len(result) > 1:
                        self._plot_progress["key"].append(key)
                        self._plot_progress["h"].append(self._h)
                        self._plot_progress["t"].append(t)
                        self._plot_progress["value"].append(result[-1])

            # early_stop
            if early_stop:
                break

            # <- inference

        # plot_progress
        if (isinstance(self._plot_progress_at, str) and self._plot_progress_at == "all") or (isinstance(self._plot_progress_at, list) and len(self._plot_progress_at) > 0 and self._h == max(self._plot_progress_at)):
            plt.close()
            sns.relplot(
                data=pd.DataFrame(self._plot_progress),
                x="t",
                y="value",
                hue="h",
                palette="rocket_r",
                col="key",
                kind='line',
                facet_kws={
                    "sharey": False,
                    "legend_out": False,
                },
            ).set(yscale='log')
            plt.draw()
            plt.pause(0.001)

        self._h += 1

        return results

    #  PRIVATE METHODS  ########################################################################################################

    def _preprocess_step_index_list(
        self,
        indices: typing.Union[str, typing.List[int]],
        T: int,
    ) -> typing.List[int]:
        """Preprocesses a specification of step indices that has been provided as an argument.

        Args:
            indices (str or list[int]): The preprocessed indices, which is either a ``str`` specification or an actual
                list of indices.

        Returns:
            list[int]: A list of integer step indices.
        """

        assert isinstance(indices, (str, list))
        assert isinstance(T, int)
        assert T > 0

        if isinstance(indices, str):  # -> indices needs to be converted

            # convert indices
            if indices == "all":
                indices = list(range(T))
            elif indices == "last":
                indices = [T - 1]
            elif indices == "last_half":
                indices = list(range(T // 2, T))
            elif indices == "never":
                indices = []
            else:
                raise NotImplementedError

        else:  # -> indices is a list already

            # ensure the indices are valid
            for t in indices:
                assert isinstance(t, int)
                assert 0 <= t < T

        return indices
