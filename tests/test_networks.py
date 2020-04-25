import pytest

from swarm import networks


def test_flat():
    networks.flat_net(2, 3)


def test_invalid_flat():
    with pytest.raises(ValueError):
        networks.flat_net(0, 3)
