import pytest
import numpy as np

from matplotlib import pyplot as plt
from swarm import animator


@pytest.fixture
def xdat():
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x - i) for i in np.linspace(0, np.pi / 2, 50)])
    z1 = np.vstack([np.cos(x - i) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    return x, dat


def test_lineswarm_kwargs(xdat):
    x, dat = xdat
    ls1 = animator.LineSwarm.auto_range(
        x, dat, set_title="Hello There", set_ylabel="loosing to a bird"
    )
    ax = plt.gca()
    # there should be no errors
    ls1.init(ax)


def test_lineswarm_kwarg_err(xdat):
    x, dat = xdat
    with pytest.raises(ValueError):
        animator.LineSwarm.auto_range(
            x, dat, set_title="Hello There", set_ylabel="loosing to a bird", doesntexist=3
        )
