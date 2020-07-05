import tempfile

import pytest
import numpy as np

from matplotlib import pyplot as plt
from swarm import animator


def test_make_animate():
    x = np.linspace(-3, 3, 10)
    y = np.sin(x)
    data = np.vstack([y + np.random.random(len(y)) for _ in range(3)])
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        print(f.name)
        animator.make_animation(x, y, data, "test", f.name)


@pytest.fixture
def xdat():
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x - i) for i in np.linspace(0, np.pi / 2, 50)])
    z1 = np.vstack([np.cos(x - i) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    return x, dat


def test_swarmanimator(xdat):
    x, dat = xdat

    ls1 = animator.LineSwarm(x, dat)
    ls2 = animator.LineSwarm(x, np.stack((z1, y1)))
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.swarm_animate([ls1, ls2], f.name)
        print(f.name)
        plt.show()


def test_lineswarm_kwargs(xdat):
    x, dat = xdat
    ls1 = animator.LineSwarm.auto_range(x, dat, set_title="Hello There",
                                        set_ylabel="loosing to a bird")
    ax = plt.gca()
    # there should be no errors
    ls1.init(ax)

def test_lineswarm_kwarg_err(xdat):
    x, dat = xdat
    with pytest.raises(ValueError):
        ls1 = animator.LineSwarm.auto_range(x, dat, set_title="Hello There",
                                        set_ylabel="loosing to a bird", doesntexist=3)





if __name__ == '__main__':
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x - i) for i in np.linspace(0, np.pi / 2, 50)])
    z1 = np.vstack([np.cos(x - i) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    ls1 = animator.LineSwarm.auto_range(x, dat)
    ls2 = animator.LineSwarm.auto_range(x, np.stack((z1, y1)))
    animator.swarm_animate([ls1, ls2], "here.mp4")
