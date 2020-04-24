#  -*- coding: utf-8 -*-

from swarm import activations


def get_activation(name):
    try:
        activfunc = getattr(activations, name)
    except AttributeError as ex:
        raise AttributeError(f"Unable to find activation {name} in torch.nn") from ex
    return activfunc
