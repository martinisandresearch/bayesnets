#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from swarm import core, regimes, activations


def main():
    x = torch.linspace(-6, 6, 101)
    y = torch.sin(x)

    static = {
        "x": x,
        "y": y,
        "num_epochs": 50,
        "activation": activations.xTanH,
    }
    ps = core.make_sweep_paramsets(static, lr=(0.001, 0.002, 0.004, 0.008))
    res = core.hive_trainer(
        regimes.default_train,
        num_swarm=4,
        fields="ypred,loss",
        paramsets=ps
    )
    from pprint import pprint
    pprint([r.keys() for r in res])


if __name__ == '__main__':
    main()
