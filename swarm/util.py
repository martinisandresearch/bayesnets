#  -*- coding: utf-8 -*-
"""
Here go the functions that are general and don't really have
code that relates to experiments, more things involving munging and stuff

Do not import anything else in swarm here. Stdlib and third party only
"""

import functools
import contextlib

import torch
from torch import nn

import numpy as np

from typing import Iterable, List, Any


def transpose(listoflists: Iterable[List[Any]]) -> List[List[Any]]:
    """
    A list of lists transpose. Say you have a structure like
    [[ time1, value1], [time2, value2], [time3, value3]]
    The function would return
    [[ time1, time2, time3], [value1, value2, value3]

    """
    return [list(i) for i in zip(*listoflists)]


@contextlib.contextmanager
def seed_as(seed: int):
    """
    We need reproducibility and this is best achieved with a seed based approach
    However, we must be very careful to not turn the seed into a hyper-parameter

    Passing in a generator is generally quite awkward in torch, though
    less so for numpy. Most docs recommend a global approach, though
    the best idea is usually to use a generator directly for all random number
    generation since it makes it explicit rather than implicit.

    In the absence of good design, we choose to make the right thing easy

    Examples:
        >>> with seed_as(10):
        ...     t = torch.rand(5)


    """
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()

    torch.manual_seed(seed)
    np.random.seed(seed)
    yield

    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)