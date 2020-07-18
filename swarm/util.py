#  -*- coding: utf-8 -*-
"""
Here go the functions that are general and don't really have
code that relates to experiments, more things involving munging and stuff

Do not import anything else in swarm here. Stdlib and third party only
"""
import functools
import contextlib

import pendulum

import torch
import numpy as np

from typing import Iterable, List, Any, Dict, Sequence, Callable


def time_me(func):
    """
    Stick this on slow functions so we can get some instant stats on it's runtime
    Examples:
        >>> @time_me
        ... def slow_func(arg):
        ...     import time
        ...     time.sleep(10)
        ...     return 3
        >>> slow_func()
        ... "Starting slow_func"
        ... "Finished in 10 seconds"
        ... 3
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(f"Starting {func.__name__}")
        start = pendulum.now()
        ret = func(*args, **kwargs)
        print(f"Finished in {(pendulum.now()-start).in_words()}")
        return ret

    return inner


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


def key_intersection(*dicts: dict) -> dict:
    """
    A function that returns the keys that overlap in the dicts. Useful to ensure
    that a merge won't overwrite values
    """
    return functools.reduce(lambda a, b: a & b.keys(), dicts)


def merge_dicts(*dicts: dict) -> dict:
    """
    Give an iterable of dicts, it merges them and checks we aren't
    overwriting a value somewhere along the way.
    """
    if key_intersection(*dicts):
        raise ValueError(f"Key intersection error {key_intersection(*dicts)}")
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def dict_slicer(dict_slice: Dict[str, Sequence]) -> Dict[str, Any]:
    """
    This function takes a dictionary of the form
    {"a": [1,2,3], "b": [4,5,6]} and checks that the values are all the same size and then
    yields a slice i.e.
    {"a": 1, "b": 4}, {"a", 2, "b": 5}, {"a", 3, "b": 6}

    While the key type doesn't matter, this is intended for use as a kwargs and should be a str
    If the data was a pandas df, this function would be like taking each row at a time.

    Yields:
        Dict[str, Any]
    """
    sizes = {len(v) for v in dict_slice.values()}
    if len(sizes) != 1:
        raise ValueError(f"Combodict {dict_slice.keys()} misconfigured with differing lengths")
    for i in range(sizes.pop()):
        yield {k: dict_slice[k][i] for k in dict_slice}
