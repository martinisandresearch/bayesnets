#  -*- coding: utf-8 -*-
"""
Building a basic hive with varying regime and architecture parameters.
"""
__author__ = "Aidan Morrison <aidandmorrison@gmail.com>"
import sys
sys.path.append('/Users/aidanmorrison/bayesnets')
import torch
from swarm import core, activations, io, regimes

import functools

def make_bee(regime, x, y, *args, **kwargs):
    """
    Convenience function for turning a simple arg
    based training function into a the bee format of argless
    This can be used to pass state between swarm iterations
    and share resources
    Also optional to use
    """
    thestrkwargs = {key: str(value) for key, value in kwargs.items()}
    return functools.partial(regime, x, y, *args, **kwargs), thestrkwargs


def main():
    x = torch.linspace(-5, 5, 61)
    y = torch.sin(x)
    seed = 10
    static_params = {'x': x.numpy(),
                   'y': y.numpy(),
                   'seed': seed}

    # run experiments for each lr and save.
    reslist = []
    for momentum in (0.7, 0.9, 0.96):
        for lr in [0.002, 0.01, 0.03]:
            swarm_dict = {}
            bee, params = make_bee(
                regimes.default_train, x, y, hidden = 1, width = 10, num_epochs=50, activation=activations.xTanH, momentum = momentum
            )
            res = core.swarm_train(bee, num_bees=4, fields="ypred,loss", seed=seed)
            swarm_dict['results'] = res
            swarm_dict['params'] = params
            reslist.append(swarm_dict.copy())
    return reslist, static_params

if __name__ == '__main__':
    main()
