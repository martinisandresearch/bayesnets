import pytest

from swarm import util


def test_transpose_1():
    """Make explicit for clarity"""
    inp = [['t1', 'v1']]
    outp = [['t1'], ['v1']]
    assert util.transpose(inp) == outp
    assert util.transpose(outp) == inp


def data_in(num):
    for i in range(num):
        yield [f'{field}{i}' for field in "t,v,u".split(",")]


def data_out(num):
    for l in 't,v,u'.split(","):
        yield [f"{l}{i}" for i in range(num)]


@pytest.mark.parametrize("size", (3, 5, 10))
def test_transpose_generator_in(size):
    assert util.transpose(data_in(size)) == list(data_out(size))


@pytest.mark.parametrize("size", (3, 5, 10))
def test_transpose_reverse(size):
    assert util.transpose(data_out(size)) == list(data_in(size))
