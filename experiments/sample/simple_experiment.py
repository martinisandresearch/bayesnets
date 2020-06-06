#  -*- coding: utf-8 -*-
"""
Minimal experiment. All we use here are the networks/activations
+ the swarm_trainer data.

Check out the example in animate_training.py which uses swarm.regimes to make training like
this trivial to define and experiment with.
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
import pendulum

from swarm import core, activations, networks


def sin_experiment():
    xt = torch.linspace(-6, 6, 100)
    yt = torch.sin(xt)
    num_epochs = 20

    net = networks.flat_net(2, 2, activations.xTanH)
    optimiser = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        st = pendulum.now()
        optimiser.zero_grad()
        ypred = net(xt)

        loss = loss_func(ypred, yt)
        et = pendulum.now() - st
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        loss.backward()
        optimiser.step()

        yield ypred, loss, et


def main():
    # this is the simplest path to finish
    results = core.swarm_train(sin_experiment, num_bees=4, fields="ypred,loss,etime", seed=10)
    return results


if __name__ == "__main__":
    main()
