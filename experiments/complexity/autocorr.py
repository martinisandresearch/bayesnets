#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging

import numpy as np
import torch
from torch import nn, optim

from swarm import networks, core, animator, activations

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


def diff(x: torch.Tensor) -> torch.Tensor:
    return x[1:] - x[:-1]


def second_deriv(ypred: torch.Tensor, mse_weight=0.01):
    fdiff = diff(ypred)
    sdiff = diff(fdiff)

    mse = ypred.norm()
    # ln2 = sdiff.abs().mean()
    ln2 = sdiff.norm()
    return mse_weight * mse - ln2


class Sin(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.sin(x)


def solo_train(
    x, hidden=2, width=2, activation=nn.ReLU, num_epochs=10, lr=0.001, momentum=0.9, corr_len=10
):
    net = networks.flat_net(hidden_depth=hidden, width=width, activation=activation)
    optimiser = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    mse_weight = 1
    loss_func = lambda ypred: second_deriv(ypred, mse_weight=mse_weight)

    for epoch in range(num_epochs):
        mse_weight = min(1 / (epoch + 1), 0.1)
        optimiser.zero_grad()
        ypred = net(x)

        loss = loss_func(ypred)
        log.debug("e: %s, loss: %s", epoch, loss)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        yield ypred, loss

        loss.backward()
        optimiser.step()


def main():
    x = torch.linspace(-10, 10, 100)
    beeparams = {
        "x": x,
        "num_epochs": 200,
        "lr": 0.005,
        "momentum": 0.5,
        "width": 50,
        "hidden": 3,
        "activation": activations.Tanh,
    }
    logging.info("Starting training")
    results = core.swarm_train(solo_train, beeparams, num_bees=20, fields="ypred,loss")
    print(results["loss"])
    log.info("Making animation")
    yd = np.zeros(len(x))
    yd[0] = -0.5
    yd[-1] = 0.5
    animator.make_animation(
        x.detach().numpy(), yd=yd, data=results["ypred"], title="secondderiv", destfile="sd.mp4"
    )


if __name__ == "__main__":
    main()
