#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from torch import nn
import pendulum

from swarm import core, activations, animator, networks, regimes


@networks.make_sequential
def dropout_flat(hidden_depth: int, width: int, activation=nn.ReLU, dropout=0.1) -> nn.Module:
    if hidden_depth < 1:
        raise ValueError("Hidden depth must be > 1")
    yield nn.Linear(1, width)
    yield activation()
    yield nn.Dropout(dropout)

    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield activation()
        yield nn.Dropout(dropout)

    yield nn.Linear(width, 1)


def bee_trainer(xt, yt, hidden=2, width=2, dropout=0.1, num_epochs=200):
    net = dropout_flat(hidden, width, activation=nn.ReLU, dropout=dropout)
    optimiser = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(xt)

        loss = loss_func(ypred, yt)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        with torch.no_grad():
            net.eval()
            y_nodrop = net(xt)
            loss_nodrop = loss_func(ypred, yt)
            net.train()

        loss.backward()
        optimiser.step()

        yield ypred, loss, y_nodrop, loss_nodrop



def main():
    xt = torch.linspace(-6, 6, 101)
    yt = torch.sin(xt)
    h, w = 2, 10
    dropout = 0.1
    nepoch = 1000
    bee = regimes.make_bee(bee_trainer, xt, yt, h, w, dropout, nepoch)
    print("Starting training")
    results = core.swarm_train(bee, num_swarm=20, fields="ypred,loss,y_nodrop,loss_nodrop")
    print("Done Training, starting animating")
    basef = f"{dropout}_dropout_{h}h{w}w_{nepoch}e"
    animator.make_animation(xt, yt, results["ypred"], basef,
                            f"data_out/{basef}.mp4")
    animator.make_animation(xt, yt, results["y_nodrop"], f"Eval {basef}",
                            f"data_out/eval_{basef}.mp4")
    print("Animation done")


if __name__ == "__main__":
    main()
