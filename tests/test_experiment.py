import pytest

import torch

import swarm
from swarm import regimes
from swarm import networks


def test_simple():
    xt = torch.linspace(-6, 6, 100)
    yt = torch.sin(xt)

    trainer = regimes.SwarmTrainerBase(
        xt,
        yt,
        lambda: networks.flat_net(2, 2, swarm.get_activation("ReLU")),
        num_epochs=10,
        loss_func=torch.nn.MSELoss(),
    )

    runner = swarm.SwarmRunner.from_string("ypred,loss", seed=10)
    results = runner.swarm_train(2, trainer.train_single)
    # print(results)
    assert results.keys() == {"ypred", "loss"}

    assert len(results["loss"]) == 2
    assert results["loss"].shape == (2, 10)
    assert results["ypred"].shape == (2, 10, 100)

    # also test seeds
    assert results["loss"][0][-1] == pytest.approx(0.5845404863357544)
