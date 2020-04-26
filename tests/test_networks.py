import pytest

from swarm import networks

import torch

def test_flat():
    networks.flat_net(2, 3)


def test_invalid_flat():
    with pytest.raises(ValueError):
        networks.flat_net(0, 3)


def test_flat_forward():
    xt = torch.linspace(-5, 5, 100)
    nw = networks.flat_net(4, 4)
    yt = nw.forward(xt)
    assert yt.size() == (100, )