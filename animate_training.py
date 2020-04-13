#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import attr
import torch
from torch import optim, nn

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
from matplotlib import animation
plt.rcParams["figure.figsize"] = (14.0, 7.0)


DEBUG = False


def make_net(hidden_depth, width):
    assert hidden_depth >= 1
    yield nn.Linear(1, width)
    yield nn.ReLU()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield nn.ReLU()
    yield nn.Linear(width, 1)


@attr.s
class Trainer:
    funcname = attr.ib(type=str)
    x_domain = attr.ib()
    loss_func = attr.ib(default=nn.MSELoss())
    optim_factory = attr.ib(default=lambda net: optim.SGD(net.parameters(), lr=0.002, momentum=0.9))

    xt = attr.ib(init=False)
    yt = attr.ib(init=False)

    def __attrs_post_init__(self):
        try:
            func = getattr(torch, self.funcname)
        except AttributeError as ex:
            raise AttributeError(f"Unable to find {self.funcname} in torch0") from ex
        xsize = (self.x_domain[1] - self.x_domain[0]) * 10 + 1
        self.xt = torch.linspace(self.x_domain[0], self.x_domain[1], xsize).unsqueeze(-1)
        self.yt = func(self.xt)

    def __str__(self):
        return f"{self.funcname}:{self.x_domain[0]}:{self.x_domain[1]}"

    def get_training_results(self, net, num_epoch):
        optimiser = self.optim_factory(net)
        data_out = torch.zeros(num_epoch, self.xt.shape[0])
        loss_t = torch.zeros(num_epoch)

        og_loss = self.loss_func(net(self.xt), self.yt)
        for epoch in range(num_epoch):
            optimiser.zero_grad()
            ypred = net(self.xt)

            loss = self.loss_func(ypred, self.yt)

            if DEBUG:
                print(epoch, loss)

            loss_t[epoch] = loss.item()
            data_out[epoch, :] = ypred.squeeze()

            loss.backward()
            optimiser.step()
        if DEBUG:
            print(f"First loss {og_loss} v final {loss}")
        return data_out.detach(), loss_t.detach()


