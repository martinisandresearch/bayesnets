#  -*- coding: utf-8 -*-


from swarm.core import SwarmRunner


def get_activation(name):
    from swarm import activations

    try:
        activfunc = getattr(activations, name)
    except AttributeError as ex:
        raise AttributeError(f"Unable to find activation {name} in activations") from ex
    return activfunc
