import pytest

import torch

import animate_training
import swarm
import swarm.core
from swarm import regimes
from swarm import networks


def test_simple():
    xt = torch.linspace(-6, 6, 100)
    yt = torch.sin(xt)

    trainer = animate_training.SwarmTrainerBase(
        xt,
        yt,
        lambda: networks.flat_net(2, 2, swarm.core.get_activation("ReLU")),
        num_epochs=10,
        loss_func=torch.nn.MSELoss(),
    )

    results = swarm.swarm_train(trainer.train_bee, 2, 10, "ypred,loss")
    # print(results)
    assert results.keys() == {"ypred", "loss"}

    assert len(results["loss"]) == 2
    assert results["loss"].shape == (2, 10)
    assert results["ypred"].shape == (2, 10, 100)

    # this tests the seed
    assert results["loss"][0][-1] == pytest.approx(0.5845404863357544)
