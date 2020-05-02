import pytest

import torch

import swarm
from swarm import core, networks, activations


def sin_experiment():
    xt = torch.linspace(-6, 6, 100)
    yt = torch.sin(xt)
    num_epochs = 10

    net = networks.flat_net(2, 2, activations.xTanH)
    optimiser = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(xt)

        loss = loss_func(ypred, yt)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        yield ypred, loss

        loss.backward()
        optimiser.step()


def test_simple():
    results = core.swarm_train(sin_experiment, 2, 10, "ypred,loss")
    # print(results)
    assert results.keys() == {"ypred", "loss"}

    assert len(results["loss"]) == 2
    assert results["loss"].shape == (2, 10)
    assert results["ypred"].shape == (2, 10, 100)

    # this tests the seed
    assert results["loss"][0][-1] == pytest.approx(0.5803773999214172)
