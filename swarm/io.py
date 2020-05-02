#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import json
import os
import pathlib
from typing import Dict, Any

import numpy as np
import pandas as pd


def to_2d(arr: np.ndarray, name: str):
    """https://stackoverflow.com/questions/46134827/
    how-to-recover-original-indices-for-a-flattened-numpy-array"""
    cols = ["swarm", "epoch"]
    if arr.ndim < 2:
        raise ValueError("All swarm data must be at minimum 2d")
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        cols.append(name)
    else:
        cols += [f"{name}_{i}" for i in range(arr.ndim - 2)]

    col_name = f"{name}_val"

    index = pd.MultiIndex.from_product([range(s) for s in arr.shape], names=cols)
    return pd.DataFrame({col_name: arr.flatten()}, index=index)


def write_data(destdir: str, metadata: Dict[str, Any], data_dict: Dict[str, np.ndarray]):
    destdir = pathlib.Path(destdir)
    with open(destdir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    for name, data in data_dict.items():
        df = to_2d(data, name)
        with open(destdir / f"{name}.csv", "w") as f:
            df.to_csv(f)


def write_data_rel_here(name, data_dict, metadata=None):
    import __main__

    if not metadata:
        metadata = {}

    rel = name
    destdir = os.path.join(os.path.dirname(__main__.__file__), "data_out", rel)
    os.makedirs(destdir, exist_ok=True)
    write_data(destdir, metadata, data_dict)
