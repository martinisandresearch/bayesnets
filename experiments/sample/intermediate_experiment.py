#  -*- coding: utf-8 -*-
"""
How to test a series of results.
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from swarm import core, activations, io, regimes


def main():
    x = torch.linspace(-6, 6, 101)
    y = torch.sin(x)

    # run experiments for each lr and save.
    for lr in (0.001, 0.002, 0.004, 0.008):

        # def default_train(x, y, hidden=2, width=2, activation=nn.ReLU, num_epochs=200, lr=0.001):
        # to compare against
        bee = regimes.make_bee(
            regimes.default_train, x, y, num_epochs=50, activation=activations.xTanH, lr=lr
        )
        res = core.swarm_train(bee, num_swarm=4, fields="ypred,loss", seed=10)
        # writing is optional, you can keep them all in memory. If you're analysing in python
        # keeping in memory is probably easier
        io.write_data_rel_here(f"intermediate_lr_{lr}", res)


if __name__ == '__main__':
    main()
