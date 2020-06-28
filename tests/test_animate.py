import os
import tempfile

import pytest
import numpy as np

from matplotlib import animation, pyplot as plt
from swarm import animator


def test_make_animate():
    x = np.linspace(-3, 3, 10)
    y = np.sin(x)
    data = np.vstack([y + np.random.random(len(y)) for _ in range(3)])
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        print(f.name)
        animator.make_animation(x, y, data, "test", f.name)


def test_swarmanimator():
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x-i) for i in np.linspace(0, np.pi/2, 50)])
    z1 = np.vstack([np.cos(x - i) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    ls1 = animator.LineSwarm(x, dat)
    ls2 = animator.LineSwarm(x, np.stack((z1, y1)))
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        animator.swarm_animate([ls1, ls2], f.name)
        print(f.name)
        plt.show()


if __name__ == '__main__':
    x = np.linspace(-3, 3, 10)
    y1 = np.vstack([np.sin(x - i) for i in np.linspace(0, np.pi / 2, 50)])
    z1 = np.vstack([np.cos(x - i) for i in np.linspace(0, np.pi, 50)])

    dat = np.stack((y1, z1))
    assert dat.shape == (2, 50, 10)
    ls1 = animator.LineSwarm(x, dat)
    ls2 = animator.LineSwarm(x, np.stack((z1, y1)))
    animator.swarm_animate([ls1, ls2], "here.mp4")

