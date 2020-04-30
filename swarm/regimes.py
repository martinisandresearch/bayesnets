#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
from typing import Callable, Any, Dict

import attr
import torch
from torch import nn
from torch.optim import sgd
import numpy as np

import swarm.core

log = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class SwarmTrainerBase:
    """
    This is a convenience class for making it easy to define a train_single
    function to pass into the core run/record code

    This is entirely optional, but provides a standard way of doing things.
    """

    xt: torch.Tensor
    yt: torch.Tensor

    network: swarm.get_network = nn.Linear
    netkwargs: Dict[str, Any] = {}

    optim: swarm.get_torch_optim = sgd.SGD
    optimkwargs: Dict[str, Any] = {}

    loss_func: swarm.get_torch_nn = nn.MSELoss()
    num_epochs: int = 200

    def __attrs_post_init__(self):
        assert self.xt.size() == self.yt.size()
        assert self.new_network()

    def to_metadata(self):
        md = attr.asdict(self)
        # remove
        del md["xt"]
        del md["yt"]

        return {
            "x": self.xt.tolist(),
            "y": self.yt.tolist(),
            "regime": self.__class__.__name__,
            "regimedict": md,
        }

    def new_network(self):
        return self.network(**self.netkwargs)

    def train_bee(self):
        net = self.new_network()
        optimiser = self.optim(net.parameters(), **self.optimkwargs)

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
