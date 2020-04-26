#  -*- coding: utf-8 -*-


import torch
from torch import nn


from typing import Any


class Squeezer(nn.Module):
    def forward(self, input: torch.Tensor, **kwargs: Any):
        return input.squeeze()


class Unsqueezer(nn.Module):
    def forward(self, input: torch.Tensor, **kwargs: Any):
        return input.unsqueeze(-1)


def make_sequential(network_maker):
    """
    Convenience wrapper - takes care of adding squeeze and unsqueeze
    so we can focus on treating 1d data in a more intuitive way
    Also converts a list/iterable of layers into a single module
    """
    def inner(*args, **kwargs):
        return nn.Sequential(Unsqueezer(), *network_maker(*args, **kwargs), Squeezer())
    return inner


@make_sequential
def flat_net(hidden_depth: int, width: int, activation=nn.ReLU) -> nn.Module:
    if hidden_depth < 1:
        raise ValueError("Hidden depth must be > 1")
    yield nn.Linear(1, width)
    yield activation()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield activation()
    yield nn.Linear(width, 1)
