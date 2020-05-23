#  -*- coding: utf-8 -*-
"""
Core.py contains the infra for the swarm work
Any changes to this file should be reviewed by @varun at the very least

Do not cross import from anything other than util
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
import random

import torch
import numpy as np
import torch.nn

from swarm import util

from typing import List, Any

log = logging.getLogger(__name__)


def condense(result: List[Any]):
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


@util.time_me
def swarm_train(bee_trainer, num_swarm=50, seed=None, fields=None):
    """
    Use this function to standardise how we run swarm training as this will take care of seeds,
    as well as the data interchange format.

    Args:
        bee_trainer: Callable[[], Generator[Tuple[Any]]]
            Takes a function that defines a full training sequence, and yields data
            every epoch. See examples and gtests
        num_swarm: int
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
    with util.seed_as(seed):
        full_res = []
        for i in range(num_swarm):
            # results can be something like ypredict, loss, epoch time.
            # they must be consistent types
            bee_result = [condense(res) for res in util.transpose(bee_trainer())]
            full_res.append(bee_result)
    nk = len(full_res[0])

    if fields:
        keys = fields.split(",")
    else:
        keys = [f"V{i + 1}" for i in range(nk)]

    # Everything is a [swarm, epoch, *dims] np.array on return
    return {k: condense(v) for k, v in zip(keys, util.transpose(full_res))}
