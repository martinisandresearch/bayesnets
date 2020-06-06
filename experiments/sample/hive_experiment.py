#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from swarm import core, regimes, activations


def hive():
    x = torch.linspace(-6, 6, 11)
    y = torch.sin(x)

    static = {"x": x, "y": y, "num_epochs": 5}

    ps = core.make_sweep_paramsets(
        static, lr=(0.001, 0.002, 0.004, 0.008), activation=[activations.xTanH, activations.ReLU]
    )

    res = core.hive_trainer(regimes.default_train, param_list=ps, num_bees=4, fields="ypred,loss")
    return res


if __name__ == "__main__":
    hive()
