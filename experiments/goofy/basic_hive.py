#  -*- coding: utf-8 -*-
"""
Building a basic hive with varying regime and architecture parameters.
"""
__author__ = "Aidan Morrison <aidandmorrison@gmail.com>"
import sys

sys.path.append("/Users/aidanmorrison/bayesnets")
sys.path.append("/Users/aidanmorrison/bayesnets/experiments/goofy")
import torch
from swarm import core, activations, io, regimes

import functools
import utils
import itertools
from torch import nn
import pandas as pd


def main(width_list=None, momentum_list=None, lr_list=None, num_epochs=100, num_bees=10, seed=10):
    """
    This is a simple experiment for allowing multiple values of width, momentum and learning rate to be tested.
    At the moment the different activations and width are not parameters, but can be varied in the function contents.
    I've had trouble calling from Reticulate with more than one hidden layer, however it seems to work fine in Python.
    """
    ### just inserting default values here, since including iterable defaults is dangerous.
    if width_list is None:
        width_list = [10]
    if momentum_list is None:
        momentum_list = [0.9]
    if lr_list is None:
        lr_list = [0.02]

    x = torch.linspace(-5, 5, 61)
    y = torch.sin(x)
    static_params = {"x": x.numpy(), "y": y.numpy(), "seed": seed}

    # run experiments for each lr and save.
    reslist = []
    hidden_list = [1]
    activations_list = [activations.xTanH, nn.ReLU, nn.Tanh]
    variations_list = [hidden_list, width_list, activations_list, momentum_list, lr_list]
    param_list = itertools.product(*variations_list)
    for params in param_list:
        swarm_dict = {}
        bee, params = utils.make_bee(
            regimes.default_train,
            x,
            y,
            hidden=params[0],
            width=params[1],
            num_epochs=num_epochs,
            activation=params[2],
            momentum=params[3],
            lr=params[4],
        )
        res = core.swarm_train(bee, num_bees=num_bees, fields="ypred,loss", seed=seed)
        swarm_dict["results"] = res
        swarm_dict["params"] = params
        reslist.append(swarm_dict.copy())
    return reslist, static_params


def get_long_results(
    width_list=None, momentum_list=None, lr_list=None, num_epochs=100, num_bees=10, seed=10
):
    """
    This is an all-in-one step that runs the above experiment, and flattens and joins it out completely into a single object.
    It is convenient for getting the data into a single flat object, particularly for saving or loading into another language like R.
    However, it's not ideal for memory consumption, as there are many repreated values in the final long table.
    """

    ### just inserting default values here, since including iterable defaults is dangerous.
    if width_list is None:
        width_list = [10]
    if momentum_list is None:
        momenutum_list = [0.9]
    if lr_list is None:
        lr_list = [0.02]
    reslist, static_params = main(width_list, momentum_list, lr_list, num_epochs, num_bees, seed)
    data, loss, xy, params = utils.unpacker(reslist, static_params)
    data_df, loss_df, xy_df, param_df = utils.make_frames(data, loss, xy, static_params, params)
    long_data = pd.melt(
        data_df, id_vars=["swarm", "bee", "epoch"], var_name="x", value_name="ypred"
    )
    long_data["x"] = long_data["x"].astype(float)
    long_data = long_data.merge(xy_df, how="left", left_on=["x", "swarm"], right_on=["x", "swarm"])
    long_data = long_data.merge(param_df, how="left", left_on="swarm", right_on="swarm")
    long_data = long_data.merge(
        loss_df, how="left", left_on=["swarm", "bee", "epoch"], right_on=["swarm", "bee", "epoch"]
    )
    return long_data


# if __name__ == '__main__':
#     main(hidden_list)
