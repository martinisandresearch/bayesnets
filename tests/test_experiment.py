import pytest

import torch

import swarm
from swarm import core, networks, activations, regimes


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
    results = core.swarm_train(sin_experiment, bee_params=None,
                               num_bees=2, seed=10, fields="ypred,loss")
    # print(results)
    assert results.keys() == {"ypred", "loss"}

    assert len(results["loss"]) == 2
    assert results["loss"].shape == (2, 10)
    assert results["ypred"].shape == (2, 10, 100)

    # this tests the seed
    assert results["loss"][0][-1] == pytest.approx(0.5803773999214172)


def test_hive():
    x = torch.linspace(-6, 6, 11)
    y = torch.sin(x)

    static = {
        "x": x,
        "y": y,
        "num_epochs": 5
    }

    ps = core.make_combo_paramsets(
        static,
        {"hidden": [1, 2, 3], "width": [4, 3, 2]},
        lr=(0.001, 0.002, 0.004, 0.008),
        activation=[activations.xTanH, activations.ReLU]
    )

    res = core.hive_trainer(
        regimes.default_train,
        param_list=ps,
        num_bees=4,
        fields="ypred,loss",
    )
    # 3 for networks, 4 for lr and 2 for activations
    assert len(res) == 3 * 4 * 2
    assert res[0]["lr"] == 0.001
    assert res[0]["activation"] == activations.xTanH
    assert all(r["num_epochs"] == 5 for r in res)
    assert res[-1]["lr"] == 0.008
    assert res[-1]["activation"] == activations.ReLU
    # checks that they do work as expected
    assert all(r["hidden"] + r["width"] == 5 for r in res)
