#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging

import numpy as np
import torch
from torch import nn, optim

from swarm import networks, core, animator

log = logging.getLogger(__name__)


def autocorr(x: torch.Tensor, n=10):
    """
    Simplified autocorrelation function
    Args:
        x:
        n:

    Returns:

    """
    # we normalise the vector so moving too far from 0 is penalised as well as too close to 0
    # otherwise the solution to minimise autocorrelation is a straight line at 0
    # and dividing by the max ends up in floating point sadness
    x = x / x.norm()
    num = 0
    for i in range(n):
        num += (x[i:] * x[: len(x) - i]).sum()
    return num


def solo_train(
    x, hidden=2, width=2, activation=nn.ReLU, num_epochs=10, lr=0.001, momentum=0.9, corr_len=10
):
    net = networks.flat_net(hidden_depth=hidden, width=width, activation=activation)
    optimiser = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    start_loss = autocorr(net(x), corr_len)
    loss = 0
    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(x)

        loss = autocorr(ypred, corr_len)
        log.debug("e: %s, loss: %s", epoch, loss)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        yield ypred, loss

        loss.backward()
        optimiser.step()

    log.debug("First loss %s v final %s", start_loss, loss)


def main():
    x = torch.linspace(-10, 10, 100)
    beeparams = {
        "x": x,
        "num_epochs": 200,
        "lr": 0.005,
        "momentum": 0.5,
        "corr_len": 20,
        "width": 2,
        "activation": nn.ReLU,
    }
    logging.info("Starting training")
    results = core.swarm_train(solo_train, beeparams, num_bees=20, fields="ypred,loss")
    print(results["loss"])
    log.info("Making animation")
    yd = np.zeros(len(x))
    yd[0] = -2
    yd[-1] = 2
    animator.make_animation(
        x.detach().numpy(), yd=yd, data=results["ypred"], title="Autocorr", destfile="autocorr.mp4"
    )


if __name__ == "__main__":
    main()
