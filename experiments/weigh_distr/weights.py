#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from torch import nn

from swarm import core, animator, networks


def bee_trainer(xt, yt, width=2, num_epochs=200):
    net = networks.flat_net(1, width, activation=nn.ReLU)

    optimiser = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(xt)

        loss = loss_func(ypred, yt)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        loss.backward()
        optimiser.step()

        weight, bias, *_ = net.parameters()
        yield ypred, weight.detach().flatten().numpy().copy(), bias.detach().numpy().copy()


def main():
    import numpy as np

    xt = torch.linspace(-3 * np.pi, 3 * np.pi, 101)
    yt = torch.sin(xt)

    bp = {"xt": xt, "yt": yt, "width": 20, "num_epochs": 4}
    # bs = list(bee_trainer(**bp))
    res = core.swarm_train(bee_trainer, bp, num_bees=5, fields="ypred,weights,biases", seed=20)
    # from pprint import pprint
    # pprint(bs)
    # print(res["weights"].shape)
    # print(res["biases"])
    # print(res["biases"].max(), res["biases"].min())
    # make_hist_animation(res["biases"], "biases")
    bw = res["biases"] / res["weights"]
    # print(bw.min(), bw.max())
    # print(np.percentile(bw, [1, 5, 90, 95]))
    # # print(bw)
    bw = bw.clip(-10, 10)

    ls = animator.LineSwarm.standard(
        xt.detach().numpy(), yt.detach().numpy(), res["ypred"][::10], set_xlim=(-10, 10)
    )
    hist = animator.HistogramSwarm.from_swarm(
        bw, 100, set_title="Biases/Weights", set_ylabel="Count", set_xlim=(-10, 10)
    )
    animator.swarm_animate([ls, hist], "weight_distr.mp4")
    # animator.swarm_animate([hist], "weights.mp4")


if __name__ == "__main__":
    main()
