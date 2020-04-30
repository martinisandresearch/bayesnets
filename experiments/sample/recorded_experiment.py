#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

from swarm import core, io, regimes
import torch


def main():
    x = torch.linspace(-6, 6, 101)
    y = torch.sin(x)

    metadata = {
        "name": "recorded_sin",
        "domain": (-6, 6),
        "seed": 10,
    }

    st = regimes.SwarmTrainerBase(
        x,
        y,
        "flat_net",
        {"hidden": 2, "width": 2, "activation": "xTanH"},
        "SGD",
        {"lr": 0.002, "momentum": 0.9},
        "MSELoss",
        num_epochs=10,
    )

    results = core.swarm_train(
        st.train_bee, num_swarm=4, fields="ypred,loss", seed=metadata["seed"]
    )
    io.write_data_rel_here({**metadata, **st.to_metadata()}, results)
