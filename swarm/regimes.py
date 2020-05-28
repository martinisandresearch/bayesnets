#  -*- coding: utf-8 -*-
"""
Do not import this in other swarm libs. This
is an end user piece of code
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
import functools

import torch
from torch import nn, optim

from swarm import activations, networks

log = logging.getLogger(__name__)


def make_bee(regime, x, y, *args, **kwargs):
    """
    Convenience function for turning a simple arg
    based training function into a the bee format of argless
    This can be used to pass state between swarm iterations
    and share resources
    Also optional to use
    """
    return functools.partial(regime, x, y, *args, **kwargs)


def default_train(x, y, hidden=2, width=2, activation=nn.ReLU, num_epochs=200, lr=0.001, momentum = 0.9):
    net = networks.flat_net(hidden_depth=hidden, width=width, activation=activation)
    loss_func = nn.MSELoss()
    optimiser = optim.SGD(net.parameters(), lr=lr, momentum = momentum)

    start_loss = loss_func(net(x), y)
    loss = 0
    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(x)

        loss = loss_func(ypred, y)
        log.debug("e: %s, loss: %s", epoch, loss)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        yield ypred, loss

        loss.backward()
        optimiser.step()

    log.debug("First loss %s v final %s", start_loss, loss)
