import pytest

import torch
import numpy as np

import swarm.io
from swarm import core


def test_condense_1dtensor():
    dt = [torch.full((1,), i) for i in range(5)]
    np.testing.assert_equal(core.condense(dt), np.arange(5, dtype=np.float))


def test_condense_2dtensor():
    dt = [torch.full((5,), i) for i in range(5)]
    nump_equiv = np.repeat(np.arange(5), 5).reshape(5, 5)
    np.testing.assert_equal(core.condense(dt), nump_equiv)


def test_condense_floats():
    dt = [float(i) for i in range(5)]
    np.testing.assert_equal(core.condense(dt), np.arange(5, dtype=np.float))


def test_condense_str():
    dt = [str(i) for i in range(5)]
    np.testing.assert_equal(core.condense(dt), np.array(dt))


def test_2dto2d():
    arr = np.arange(3 * 5).reshape(3, 5)
    df = swarm.io.to_2d(arr, "ypred")
    assert df.loc[(0, 0)].ypred_val == 0
    assert df.loc[(1, 2)].ypred_val == 7
    assert df.loc[(2, 4)].ypred_val == 14


def test_3dto2d():
    arr = np.arange(3 * 5 * 7).reshape(3, 5, 7)
    df = swarm.io.to_2d(arr, "ypred")
    assert df.loc[(0, 0, 0)].ypred_val == 0
    assert df.loc[(1, 2, 3)].ypred_val == 52
    assert df.loc[(2, 4, 6)].ypred_val == 104


def test_4dto2d():
    arr = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    df = swarm.io.to_2d(arr, "ypred")
    # print(df.head())
    # print(df.index)
    assert df.loc[(0, 0, 0, 0)].ypred_val == 0
    assert df.loc[(1, 2, 3, 4)].ypred_val == 200
    assert df.loc[(1, 2, 4, 6)].ypred_val == 209
