#  -*- coding: utf-8 -*-

from torch import nn

from swarm import util


@util.collector(nn.Sequential)
def flat_net(hidden_depth, width, activation=nn.ReLU):
    assert hidden_depth >= 1
    yield nn.Linear(1, width)
    yield activation()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield activation()
    yield nn.Linear(width, 1)
