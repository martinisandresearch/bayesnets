#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
from typing import Callable, Any

import attr
import torch
from torch import nn
from torch.optim import sgd
import numpy as np

log = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class SwarmTrainerBase:
    """
    This is a convenience class for making it easy to define a train_single
    function to pass into the core run/record code

    This is entirely optional
    """

    xt: torch.Tensor
    yt: torch.Tensor

    net_factory: Callable[[], nn.Module] = lambda: nn.Linear(4, 4)
    num_epochs: int = 200

    loss_func: nn.Module = nn.MSELoss()
    optimiser: Callable = lambda netp: sgd.SGD(netp, lr=0.002, momentum=0.9)

    def __attrs_post_init__(self):
        assert self.xt.size() == self.yt.size()

    def train_single(self):
        net = self.net_factory()
        optimiser = self.optimiser(net.parameters())

        start_loss = self.loss_func(net(self.xt), self.yt)
        loss = 0
        for epoch in range(self.num_epochs):
            optimiser.zero_grad()
            ypred = net(self.xt)

            loss = self.loss_func(ypred, self.yt)
            log.debug("e: %s, loss: %s", epoch, loss)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss, poorly configured experiment")

            yield ypred, loss

            loss.backward()
            optimiser.step()

        log.debug("First loss %s v final %s", start_loss, loss)
