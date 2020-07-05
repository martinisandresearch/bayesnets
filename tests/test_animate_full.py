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
    z1 = np.vstack([np.cos((x - i) / 2) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    return x, dat


def test_swarmanimator(xdat):
    x, dat = xdat
    dat_flip = np.flip(dat, 0)

    ls1 = animator.LineSwarm(x, dat)
    ls2 = animator.LineSwarm(x, dat_flip)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.swarm_animate([ls1, ls2], f.name)
        print(f.name)
        plt.show()


def test_hist_animation():
    nepoch = 20
    x = np.random.normal(np.arange(-5, 5, nepoch), size=(nepoch, 100))
    y = np.random.normal(scale=np.arange(1, 5, nepoch), size=(nepoch, 100))
    histx = animator.HistogramSwarm(x, 10)
    histy = animator.HistogramSwarm(y, 10)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.swarm_animate([histx, histy], f.name)
        plt.show()


def test_line_hist_animation(xdat):
    nepoch = 50
    x = np.random.normal(np.arange(-5, 5, nepoch), size=(nepoch, 100))
    histx = animator.HistogramSwarm(x, 10)

    y, dat = xdat
    ls1 = animator.LineSwarm(y, dat)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.swarm_animate([histx, ls1], f.name)
        plt.show()


def manual_run():
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x - i) for i in np.linspace(0, np.pi / 2, 50)])
    z1 = np.vstack([np.cos((x - i) * 2) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    ls1 = animator.LineSwarm.auto_range(x, dat)
    ls2 = animator.LineSwarm.auto_range(x, np.stack((z1, y1)))
    animator.swarm_animate([ls1, ls2], "lines.mp4")

    nepoch = 100
    x = np.random.normal(np.arange(-10, 10, nepoch), size=(nepoch, 100))
    y = np.random.lognormal(mean=np.arange(1, 5, nepoch), size=(nepoch, 100))
    histx = animator.HistogramSwarm(x, 10)
    histy = animator.HistogramSwarm(y, 10)
    animator.swarm_animate([histx, histy], "hist.mp4")
