#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from torch import nn, optim

from swarm import animator, core, networks, regimes


class PairedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        assert 1 in (in_features, out_features)
        assert in_features * out_features % 2 == 0

    def dist_cost(self):
        # print()
        # print(self.weight.shape)

        if self.weight.shape[1] == 1:
            evens = self.weight[::2, 0]
            odds = self.weight[1::2, 0]

        else:
            evens = self.weight[0, ::2]
            odds = self.weight[0, 1::2]

        # print(evens)
        # print(odds)
        return ((evens - odds) ** 2).sum()


class AltSumLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        assert in_features % 2 == 0
        super().__init__(in_features // 2, out_features, bias)

    def forward(self, x):
        odds = x[:, 1::2]
        evens = x[:, ::2]
        # print(evens.shape, self.weight.shape, self.bias.shape)
        return (evens - odds) @ self.weight.T + self.bias


@networks.make_sequential
def make_pair_relu_network(width=10):
    yield PairedLinear(1, width)
    yield nn.ReLU()
    yield AltSumLayer(width, 1)


def dist_cost(network):
    cost = 0
    for layer in network:
        if isinstance(layer, PairedLinear):
            cost += layer.dist_cost()
    # print(f"cost is {cost}")
    return cost


def to_np(tensor):
    return tensor.detach().numpy()


def paired_relu_train(x, y, width=10, num_epochs=200, lr=0.001, momentum=0.9):
    net = make_pair_relu_network(width=width)
    loss_func = nn.MSELoss()
    optimiser = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(x)

        dc = dist_cost(net)
        loss = loss_func(ypred, y) + 10 * dc
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        yield ypred, loss, dc

        loss.backward()
        optimiser.step()
    # print(f"Weight, {net[1].weight}")
    # print(f"bias: {net[1].bias}")
    # print(net[3].weight.T, net[3].bias)


def main():
    import numpy as np

    xt = torch.linspace(-1, 1.5, 101)
    yt = torch.sin(xt * np.pi) + 0.2 * torch.sin(10 * np.pi * xt)

    bp = {"x": xt, "y": yt, "width": 20, "num_epochs": 200, "lr": 0.02}
    results = core.swarm_train(paired_relu_train, bp, num_bees=10, fields="ypred,loss,penalty")
    # print(results['penalty'][-1])

    bp["lr"] = 0.01
    bp["width"] = 10
    bp["hidden"] = 1
    bp["activation"] = nn.ReLU

    results_relu = core.swarm_train(regimes.default_train, bp, num_bees=10, fields="ypred,loss")

    bp["hidden"] = 1
    bp["activation"] = nn.Tanh

    results_tanh = core.swarm_train(regimes.default_train, bp, num_bees=10, fields="ypred,loss")

    xn = xt.detach().numpy()
    yn = yt.detach().numpy()

    ls = animator.LineSwarm.standard(xn, yn, results["ypred"], set_title="Pair ReLu")

    ls_relu = animator.LineSwarm.standard(xn, yn, results_relu["ypred"], set_title="ReLU")
    ls_tanh = animator.LineSwarm.standard(xn, yn, results_tanh["ypred"], set_title="TanH")
    animator.swarm_animate([ls, ls_relu, ls_tanh], "pair_relu.mp4")


if __name__ == "__main__":
    main()


def test():
    l = PairedLinear(1, 10)
    l2 = PairedLinear(10, 1)
    print(l.weight.shape)
    print(l.dist_cost())
    print(l2.dist_cost())
