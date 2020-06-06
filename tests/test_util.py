import pytest

from swarm import util


def test_transpose_1():
    """Make explicit for clarity"""
    inp = [["t1", "v1"]]
    outp = [["t1"], ["v1"]]
    assert util.transpose(inp) == outp
    assert util.transpose(outp) == inp


def data_in(num):
    for i in range(num):
        yield [f"{field}{i}" for field in "t,v,u".split(",")]


def data_out(num):
    for l in "t,v,u".split(","):
        yield [f"{l}{i}" for i in range(num)]


@pytest.mark.parametrize("size", (3, 5, 10))
def test_transpose_generator_in(size):
    assert util.transpose(data_in(size)) == list(data_out(size))


@pytest.mark.parametrize("size", (3, 5, 10))
def test_transpose_reverse(size):
    assert util.transpose(data_out(size)) == list(data_in(size))


def test_merges():
    d1 = {"a": 3}
    d2 = {"b": 4}
    d3 = {"c": 5, "d": 6}
    assert util.merge_dicts(d1, d2, d3) == {"a": 3, "b": 4, "c": 5, "d": 6}


def test_combo_conv():
    mp = list(util.dict_slicer({"b": (1, 2, 3), "c": (4, 5, 6)}))
    assert mp == [{"b": i, "c": j} for i, j in zip((1, 2, 3), (4, 5, 6))]
