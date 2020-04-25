#  -*- coding: utf-8 -*-
"""
Core.py contains the infra for the swarm work
Any changes to this file should be reviewed by @varun at the very least

Do not cross import from anything other than util
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging

import attr
import torch
from torch import nn
from torch.optim import optimizer, sgd

from animate_training import DEBUG


log = logging.getLogger(__name__)

@attr.s(auto_attribs=True)
class SwarmTrainer:
    xt: torch.Tensor
    yt: torch.Tensor

    loss_func: nn.Module = nn.MSELoss()
    optimfunc: optimizer.Optimizer = sgd.SGD
    optimkwargs: dict = attr.Factory(lambda: {"lr": 0.002, "momentum": 0.9})

    def __str__(self):
        xm = round(self.xt.min().item(), 2), round(self.xt.max().item(), 2)
        domainstr = f"[{xm[0]}:{xm[1]}]"
        return f"{domainstr}"

    @property
    def optimiser(self):
        return lambda netp: self.optimfunc(netp, **self.optimkwargs)

    def get_training_results(self, net, num_epoch):
        optimiser = self.optimiser(net.parameters())
        data_out = torch.zeros(num_epoch, self.xt.shape[0])
        loss_t = torch.zeros(num_epoch)

        start_loss = self.loss_func(net(self.xt), self.yt)
        loss = 0
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
            print(f"First loss {start_loss} v final {loss}")
        return data_out.detach(), loss_t.detach()