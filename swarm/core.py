#  -*- coding: utf-8 -*-
"""
Core.py contains the infra for the swarm work
Any changes to this file should be reviewed by @varun at the very least

Do not cross import from anything other than util
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
import random

import attr


import torch
from torch import nn
from torch.optim import optimizer, sgd
import numpy as np

from swarm import util

from typing import List, Any

log = logging.getLogger(__name__)


def condense(result: List[Any]):
    """
    We assume that the results can be converted into a np.array
    Since these are the base types of all things in scientific python
    """
    firstel = result[0]
    if isinstance(firstel, torch.Tensor):
        if len(firstel) == 1:
            return np.array([el.item() for el in result])
        return torch.stack(result).detach().numpy()
    else:
        pass





@attr.s
class SwarmLogger:
    fields = attr.ib(type=list)
    seed = attr.ib(type=int, default=random.randint(0, 2 ** 31))

    # ddict = attr.ib(init=False, default={k: [] for k in fields})

    @classmethod
    def from_string(cls, field_str: str):
        fields = field_str.split(",")
        return cls(fields)

    def swarm_train(self, num_swarm, trainer_factory):
        ddict = {k: [] for k in self.fields}
        with util.seed_as(self.seed):
            for i in range(num_swarm):
                # results can be something like ypredict, loss, epoch time.
                # they must be consistent types
                results = util.transpose(trainer_factory())


@attr.s(auto_attribs=True)
class SwarmTrainerBase:
    xt: torch.Tensor
    yt: torch.Tensor

    loss_func: nn.Module = nn.MSELoss()
    optimfunc: optimizer.Optimizer = sgd.SGD
    optimkwargs: dict = attr.Factory(lambda: {"lr": 0.002, "momentum": 0.9})

    @property
    def optimiser(self):
        return lambda netp: self.optimfunc(netp, **self.optimkwargs)

    def train_single(self, net, num_epoch):
        optimiser = self.optimiser(net.parameters())
        data_out = torch.zeros(num_epoch, self.xt.shape[0])
        loss_t = torch.zeros(num_epoch)

        start_loss = self.loss_func(net(self.xt), self.yt)
        loss = 0
        for epoch in range(num_epoch):
            optimiser.zero_grad()
            ypred = net(self.xt)

            loss = self.loss_func(ypred, self.yt)

            log.debug("e: %s, loss: %s", epoch, loss)

            loss_t[epoch] = loss.item()
            data_out[epoch, :] = ypred.squeeze()
            yield ypred.squeeze(), loss.item()

            loss.backward()
            optimiser.step()
        log.debug("First loss %s v final %s", start_loss, loss)
        return data_out.detach(), loss_t.detach()

    def train_swarm(self, network_factory, swarm_size, num_epoch):
        for i in range(swarm_size):
            net = network_factory()
            traindata, loss = self.train_single(net, num_epoch)
