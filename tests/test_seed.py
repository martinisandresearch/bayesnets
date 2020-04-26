import pytest

import torch
import numpy as np

from swarm import util


@pytest.mark.parametrize("seed,ans", (
        (5, 11),
        (100, 40),
        (95, 10)
)
                         )
def test_torch_seed(seed, ans):
    with util.seed_as(seed):
        a = torch.randint(100, (1,))
    assert a.item() == ans


@pytest.mark.parametrize("seed,ans", (
        (5, 99),
        (100, 8),
        (95, 22)
)
                         )
def test_np_seed(seed, ans):
    with util.seed_as(seed):
        a = np.random.randint(100)
    assert a == ans


def test_nested():
    with util.seed_as(5):
        a = np.random.randint(100)
        with util.seed_as(100):
            b = np.random.randint(100)
        c = np.random.randint(100)
    assert a == 99
    assert b == 8
    assert c == 78


def test_nested_dub():
    """paired test with the above"""
    with util.seed_as(5):
        a = np.random.randint(100)
        with util.seed_as(100):
            b = np.random.randint(100)
            d = np.random.randint(100)
        c = np.random.randint(100)
    assert a == 99
    assert b == 8
    assert c == 78
    assert d == 24