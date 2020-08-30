#  -*- coding: utf-8 -*-
"""
Core.py contains the infra for the swarm work
Any changes to this file should be reviewed by @varun at the very least

Do not cross import from anything other than util
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import itertools
import logging
import random
import sys

import torch
import numpy as np
import torch.nn
import tqdm

from swarm import util

from typing import List, Any, Dict, Sequence, Iterable, Callable


log = logging.getLogger(__name__)


def condense(result: List[Any]) -> np.ndarray:
    """
    We assume that the results can be converted into a np.array
    Since these are the base types of all things in scientific python
    and pandas and it can handle all types
    """
    firstel = result[0]
    if isinstance(firstel, torch.Tensor):
        try:
            return np.array([el.item() for el in result])
        except ValueError:
            return torch.stack(result).detach().numpy()
    elif isinstance(firstel, np.ndarray):
        return np.stack(result)
    else:
        return np.array(result)


def swarm_train(
    bee_trainer, bee_params=None, num_bees=50, seed=None, fields=None
) -> Dict[str, Any]:
    """
    Use this function to standardise how we run swarm training as this will take care of seeds,
    as well as the data interchange format.

    Args:
        bee_trainer:
            Takes a function that defines a full training sequence, and yields data
            every epoch. See examples and tests
        bee_params: Dict[str, Any]
            Input parameters to the bee trainer. Defaults to nothing
        num_bees: int
            Number of swarms to run. Runtime scales linearly with this
        seed: int
            Reproduciblity hinges on this. No seed results in a random seed
        fields: str
            in the form of a comma separated string "ypred,loss,epoch_time"

    Returns:
        Dict[str, np.ndarray]
        Will accrue the bee_trainer's yields into numpy.ndarray with names given as per field
        The shape will be [swarm, epoch, *data_dim].
        For example,
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    if not bee_params:
        bee_params = {}

    with util.seed_as(seed):
        full_res = []
        for _ in tqdm.trange(num_bees, desc="swarm", disable=None):
            # results can be something like ypredict, loss, epoch time.
            # they must be consistent types
            bee_result = [condense(res) for res in util.transpose(bee_trainer(**bee_params))]
            full_res.append(bee_result)
    nk = len(full_res[0])

    if fields:
        keys = fields.split(",")
    else:
        keys = [f"V{i + 1}" for i in range(nk)]

    # Everything is a [swarm, epoch, *dims] np.array on return
    return {k: condense(v) for k, v in zip(keys, util.transpose(full_res))}


def make_sweep_paramsets(static: Dict[str, Any], **kwargs: Sequence):
    """
    A simplified version of make_combo_paramsets when you assume none of the
    parameters are dependent on each other.

    This is equivalent to ``make_combo_paramsets(static, {}, **kwargs)``

    Yields:
        Dict[str, Any]

    See Also:
        make_combo_paramsets

    """
    for dynamic in itertools.product(*kwargs.values()):
        dync = {k: v for k, v in zip(kwargs, dynamic)}
        yield util.merge_dicts(static, dync)


def make_combo_paramsets(static: Dict[str, Any], *combo: Dict[str, Sequence], **kwargs: Sequence):
    """
    A function for building a paramset for hive-style training. For a bee, we may have a series
    of parameters we're interested in and want to generate data across a sweep of various params

    These dictionaries are used by the training procedure to configure the bee as passed in
    via **kwargs

    Args:
        static: dictionary of static params that do not change
        *combo: A way to configure the params to sweep across.
            Formed by {"param": [p1, p2, ... pn], "other": [o1, o2, .., on]}
        **kwargs: A convenience method for single vars instead of doing
        {"lr": (0.001, 0.002, 0.004)} you can simply do
        lr = (0.001, 0.002, 0.004) in the function call.

    Yields:
        Dict[str, Any] - the params to feed into a bee as **kwargs

    Examples:
        See hive_experiment.py
        >>> pm = make_combo_paramsets({"x": 3}, \
            {"lr": (0.001, 0.004), "momentum": (.9, .95)}, \
            activation=(torch.nn.Sigmoid, torch.nn.ReLU))
        >>> for p in pm:
        ...    print(p)
        {"x": 3, "lr": 0.001, "momentum": .9, "activation":torch.nn.Sigmoid}
        {"x": 3, "lr": 0.004, "momentum": .95, "activation":torch.nn.Sigmoid}
        {"x": 3, "lr": 0.001, "momentum": .9, "activation":torch.nn.ReLU}
        {"x": 3, "lr": 0.004, "momentum": .95, "activation":torch.nn.ReLU}

        Note that lr and momentum are paired while activation is independent.
    """
    # remove empty dicts
    combo = [c for c in combo if c]
    if not combo:
        yield from make_sweep_paramsets(static, **kwargs)
    else:
        for dyn_combo in itertools.product(*(util.dict_slicer(combodict) for combodict in combo)):
            newst = util.merge_dicts(static, *dyn_combo)
            yield from make_sweep_paramsets(newst, **kwargs)


def hive_trainer(
    bee: Callable,
    param_list: Iterable[Dict[str, Any]],
    num_bees: int = 50,
    seed: int = None,
    fields: str = None,
) -> List[Dict[str, Any]]:
    """
    This extends swarm_train to do a sweep across parameter_list.

    Args:
        bee: Callable
        param_list: An iterable of kwargs to be passed to the bee

    Returns:
        List[Dict[str, Any]] - Tidy-esque data containing both the parameter and the bee result
        Will look like [{"lr": 0.01, "momentum": 0.95, "ypred": np.array, "loss", np.array},
                         "lr": 0.01, "momentum": 0.9, "ypred": np.array, "loss", np.array]]
        You can call pd.DataFrame on this result to turn it into a dataframe.

    """
    ret = []
    for param in param_list:
        res = swarm_train(bee, param, num_bees, seed, fields)
        ret.append({**param, **res})
    return ret
