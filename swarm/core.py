#  -*- coding: utf-8 -*-
"""
Core.py contains the infra for the swarm work
Any changes to this file should be reviewed by @varun at the very least

Do not cross import from anything other than util
"""
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import logging
import random

import attr

import torch
import numpy as np

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


@attr.s(auto_attribs=True)
class SwarmRunner:
    fields: List[str]  # this doesn't actually need to know, perhaps we can just leave it implicit?
    seed: int = random.randint(0, 2 ** 31)

    @classmethod
    def from_string(cls, field_str: str, *args, **kwargs):
        fields = field_str.split(",")
        return cls(fields, *args, **kwargs)

    def swarm_train(self, num_swarm, bee_trainer):
        ddict = {k: [] for k in self.fields}
        with util.seed_as(self.seed):
            for i in range(num_swarm):
                # results can be something like ypredict, loss, epoch time.
                # they must be consistent types
                results = util.transpose(bee_trainer())
                for field, res in zip(self.fields, results):
                    ddict[field].append(condense(res))
        # Everything is a [swarm, epoch, *dims] np.array on return
        return {k: condense(v) for k, v in ddict.items()}


# TODO: define writer's and readers?
# Will need metadata recorded?
