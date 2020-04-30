#  -*- coding: utf-8 -*-


# from swarm.core import get_activation, get_loss, get_network, get_regime


def _get_from_module(module, name):
    if callable(name):
        # assume this is the function requested
        return name
    try:
        activfunc = getattr(module, name)
    except AttributeError as ex:
        raise AttributeError(f"Unable to find {name} in {module}") from ex
    return activfunc


def get_activation(name):
    import swarm.activations

    return _get_from_module(swarm.activations, name)


def get_network(name):
    import swarm.networks

    return _get_from_module(swarm.networks, name)


def get_regime(name):
    import swarm.regimes

    return _get_from_module(swarm.regimes, name)


def get_torch_nn(name):
    import torch.nn

    return _get_from_module(torch.nn, name)


def get_torch_optim(name):
    import torch.nn

    return _get_from_module(torch.optim, name)
