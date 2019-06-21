"""
Implement the OneCycle LR schedule. Inspired by
https://github.com/sgugger/Adam-experiments.git
"""

from math import cos, pi
from typing import Dict, Sequence, Tuple, Union

from torch.optim.optimizer import Optimizer


class TrainingPhase:
    """Represents a training phase for a single parameter."""
    def __init__(self, length: float, start: float, end: float = None):
        """
        :param length: the length of the phase in number of epochs.
        :param start: the starting value of the parameter scheduled.
        :param end: the ending value of the parameter scheduled.
        """
        self.length = length
        self.start = start
        self.end = end if end is not None else start

    def __call__(self, epoch: float) -> float:
        """
        Returns the value of the parameter at ``epoch``.

        :param epoch: the epoch the training is at *in this phase* (i.e. this
                      function should not be called with the universal epoch
                      count).
        :returns: the value of the parameter.
        """
        raise NotImplementedError('__call__ must be implemented.')


class ConstantPhase(TrainingPhase):
    """A :class:`TrainingPhase` that keeps the parameter constant."""
    def __init__(self, length: float, start: float, end: float = None):
        super().__init__(length, start, end)
        if self.start != self.end:
            raise ValueError('For {}, start must be equal to end.'.format(
                self.__class__.__name__))

    def __call__(self, epoch: float) -> float:
        return self.start


class LinearPhase(TrainingPhase):
    """Linear interpolation between the start and end values."""
    def __call__(self, epoch: float) -> float:
        return (epoch / self.length) * (self.end - self.start) + self.start


class CosinePhase(TrainingPhase):
    """Cosine interpolation (?) between the start and end values."""
    def __init__(self, length: float, start: float, end: float = None):
        super().__init__(length, start, end)
        self.avg = (self.start + self.end) / 2
        self.navg = (self.start - self.end) / 2

    def __call__(self, epoch: float) -> float:
        return cos(epoch / self.length * pi) * self.navg + self.avg


class ParamScheduler:
    def __init__(self, optimizer: Optimizer,
                 param_phases: Dict[Union[str, Tuple[str, int]],
                                    Sequence[TrainingPhase]]):
        """
        :paaram optimizer: the optimizer to schedule.
        :param param_phases: maps parameters to the list of training phases
                             they should go through. A parameter can either be
                             defined by its name, or a its name and an index
                             (for the case when it is part of a tuple, e.g.
                             ``betas`` in :class:`torch.optim.Adam`).
        """
        self.epoch = -1
        self.optimizer = optimizer
        self.param_phases = param_phases

    def new_epoch(self):
        """Increases the epoch counter by one."""
        self.epoch += 1

    def update(self, epoch_decimal: float):
        """
        Updates the values of all scheduled parameters.

        :param epoch_decimal: the decimal part of the epoch count. Since it is
                              the user who knows this information (e.g. because
                              of variable batch sizes), they have to support it
                              to the function.
        """
        epoch = self.epoch + epoch_decimal
        for param, phases in self.param_phases.items():
            value = self._get_value(phases, epoch)
            pg = self.optimizer.param_groups[0]
            if isinstance(param, tuple):
                param_list = list(pg[param[0]])
                param_list[param[1]] = value
                pg[param[0]] = tuple(param_list)
            else:
                pg[param] = value

    def _get_value(self, phases: Sequence[TrainingPhase], epoch: float) -> float:
        """
        Finds the training phase that corresponds to ``epoch`` and calls it to
        get the value of the parameter.

        :param phases: a sequence of phases.
        :param epoch: the current (decimal) epoch.
        :returns: the value of the parameter.
        """
        for phase in phases:
            if phase.length > epoch:
                return phase(epoch)
            else:
                epoch -= phase.length
        else:
            # The last phase never ends; or should this be an error?
            return phase(phase.length)
