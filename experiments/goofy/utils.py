import functools
import numpy as np
import pandas as pd


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
