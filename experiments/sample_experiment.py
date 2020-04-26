#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from torch import optim

import swarm
from swarm import regimes
from swarm import networks


def main():
    xt = torch.linspace(-6, 6, 100)
    yt = torch.sin(xt)

    # this object is kind of optional, it just provides an easy
    # way of doing simple experiments easily.
    trainobj = regimes.SwarmTrainerBase(
        xt,
        yt,
        lambda: networks.flat_net(2, 2, swarm.get_activation("RelU")),
        num_epochs=200,
        loss_func=torch.nn.MSELoss(),
        optimiser=lambda x: optim.SGD(x, lr=0.002, momentum=0.9)
    )

    # This a core object
    # it takes a list of parameters being recorded and a seed
    runner = swarm.SwarmRunner.from_string("ypred,loss", seed=10)
    # the run requires a function with no args that will do training from scratch
    # and
    results = runner.swarm_train(50, trainobj.train_single)
    print(results)


