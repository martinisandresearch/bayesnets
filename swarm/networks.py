#  -*- coding: utf-8 -*-

from torch import nn

from swarm import util


@util.collector(lambda x: nn.Sequential(*x))
def flat_net(hidden_depth : int, width: int, activation=nn.ReLU) -> nn.Module:
    if hidden_depth < 1:
        raise ValueError("Hidden depth must be > 1")
    yield nn.Linear(1, width)
    yield activation()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield activation()
    yield nn.Linear(width, 1)
