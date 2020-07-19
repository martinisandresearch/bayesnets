#  -*- coding: utf-8 -*-
"""
Building a basic hive with varying regime and architecture parameters.
"""
__author__ = "Aidan Morrison <aidandmorrison@gmail.com>"

import functools
import itertools

import numpy as np
import pandas as pd

import torch
from torch import nn

from swarm import core, activations, io, regimes


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
        bee, params = make_bee(
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
    data, loss, xy, params = unpacker(reslist, static_params)
    data_df, loss_df, xy_df, param_df = make_frames(data, loss, xy, static_params, params)
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
def make_bee(regime, x, y, *args, **kwargs):
    """
    Convenience function for turning a simple arg
    based training function into a the bee format of argless
    This can be used to pass state between swarm iterations
    and share resources
    Also optional to use
    """
    thestrkwargs = {key: str(value) for key, value in kwargs.items()}
    return functools.partial(regime, x, y, *args, **kwargs), thestrkwargs


def add_anindex(an_array: np.ndarray, anindex):
    """
    This is a crude way of adding a column onto a numpy array populated with a constant.
    It just allows us to stay with numpy while we join multiple arrays
    together while we reduce dimensions.
    """
    return np.c_[an_array, np.array([anindex for a in range(len(an_array))])]


def flatten_sim(data_list, loss_list, xd, yd):
    """
    This is adds indices to lists of arrays to return single stacked
    arrays using an new col in each array carrying the index.
    This is currently limited to simulations of equal domain
    and target function (i.e. x and y is same). Could use a refactor to expand.
    """
    xy = np.c_[xd, yd]
    data_list = [add_anindex(data_list[a], a) for a in range(len(data_list))]
    flat_data = data_list[0]
    for i in range(1, len(data_list)):
        flat_data = np.vstack((flat_data, data_list[i]))
    loss_list = [add_anindex(loss_list[a], a) for a in range(len(loss_list))]
    flat_loss = loss_list[0]
    for i in range(1, len(loss_list)):
        flat_loss = np.vstack((flat_loss, loss_list[i]))

    return flat_data, flat_loss, xy


def unpacker(reslist: list, static_params: dict):
    """
    This takes the whole listed output from a hive experiment (i.e multiple swarms)
    and flattens it into four dataframes, with indices to note swarm.
    Currently assumes that x and y are static parameters and don't change across
    the entire hive, but is intended to expand to allow that to adjust.
    This function leaves data in numpy arrays and lists of dicts. No pandas required.
    """
    data_list = [x["results"]["ypred"] for x in reslist]
    loss_list = [x["results"]["loss"] for x in reslist]
    params_list = [x["params"] for x in reslist]
    params_list = [params_list[i].update({"swarm": i}) for i in range(len(params_list))]
    xd = static_params["x"]
    yd = static_params["y"]
    all_data = []
    all_loss = []
    all_xy = []
    for i in range(len(reslist)):
        flat_data, flat_loss, xy = flatten_sim(data_list[i], loss_list[i], xd, yd)
        flat_data = add_anindex(flat_data, i)
        flat_loss = add_anindex(flat_loss, i)
        xy = add_anindex(xy, i)
        params = reslist[i]["params"]
        if i == 0:
            all_data = flat_data
            all_loss = flat_loss
            all_xy = xy
            all_params = [params]
        else:
            all_data = np.vstack([all_data, flat_data])
            all_loss = np.vstack([all_loss, flat_loss])
            all_xy = np.vstack([all_xy, xy])
            all_params.append(params)
    return all_data, all_loss, all_xy, all_params


def make_frames(
    data: np.ndarray, loss: np.ndarray, xy: np.ndarray, static_params: dict, params: list
):
    """
    This function takes numpy arrays and lists of dicts and turns them into pandas dataframes,
    including hopefully clear column names.
    """
    data_df = pd.DataFrame(data, columns=[a for a in static_params["x"]] + ["bee", "swarm"])
    epoch_vec = []
    epochs = int(data_df.shape[0] / (len(data_df["bee"].unique()) * len(data_df["swarm"].unique())))

    # creating a suitable vector to note the epoch for each bee
    for i in range((len(data_df["bee"].unique()) * len(data_df["swarm"].unique()))):
        epoch_vec = epoch_vec + list(range(epochs))

    # binding on the epochs, making dfs
    data_df["epoch"] = epoch_vec
    loss_df = pd.DataFrame(loss, columns=["loss", "bee", "swarm"])
    xy_df = pd.DataFrame(xy, columns=["x", "y", "swarm"])
    loss_df["epoch"] = epoch_vec
    param_df = pd.DataFrame(params)

    return data_df, loss_df, xy_df, param_df
