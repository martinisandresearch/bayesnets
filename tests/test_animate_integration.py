#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import tempfile
import pytest
import torch
from torch import nn

from swarm import core, regimes, animator


@pytest.fixture
def hive_data():
    # define the x and y
    x = torch.linspace(-5, 5, 61)
    y = torch.sin(x)
    # define the neural network training params that won't change
    static = {"x": x, "y": y, "hidden": 1, "momentum": 0.9, "lr": 0.01, "num_epochs": 10}
    pm = core.make_combo_paramsets(
        static,
        # sweep across width
        width=[1, 3, 10],
        # and activations
        activation=[nn.ReLU, nn.Tanh, nn.Tanhshrink],
    )
    # give us the results
    res = core.hive_trainer(
        regimes.default_train, param_list=pm, num_bees=5, seed=10, fields="ypred,loss"
    )
    for s in res:
        s["activation"] = s["activation"].__name__
    return res


@pytest.mark.parametrize("epoch", (0, 9))
def test_hive_plot(hive_data, epoch):
    ans = animator.hive_plot(hive_data, "width", "activation", epoch)


def test_hive_animate(hive_data):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.hive_animate(hive_data, "width", "activation", f.name)
