import pytest

import torch
import numpy as np

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
