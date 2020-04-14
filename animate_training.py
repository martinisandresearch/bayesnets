#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os

import attr
import click
import pendulum
import numpy as np

import torch
from torch import nn
from torch.optim import optimizer, sgd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

plt.rcParams["figure.figsize"] = (14.0, 7.0)
sns.set()

DEBUG = False


def get_activation(activation):
    try:
        activfunc = getattr(nn, activation)
    except AttributeError as ex:
        raise AttributeError(f"Unable to find activation {activation} in torch.nn") from ex
    return activfunc


def make_net(hidden_depth, width, activation=nn.ReLU):
    assert hidden_depth >= 1
    yield nn.Linear(1, width)
    yield activation()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield activation()
    yield nn.Linear(width, 1)


@attr.s(auto_attribs=True)
class Trainer:
    funcname: str  # a name for the function approximated
    xt: torch.Tensor
    yt: torch.Tensor

    loss_func: nn.Module = nn.MSELoss()
    optimfunc: optimizer.Optimizer = sgd.SGD
    optimkwargs: dict = attr.Factory(lambda: {"lr": 0.002, "momentum": 0.9})

    @classmethod
    def from_function(cls, funcname, xt):
        try:
            func = getattr(torch, funcname)
        except AttributeError as ex:
            raise AttributeError(f"Unable to find {funcname} in torch0") from ex
        return cls(funcname, xt, func(xt))

    @classmethod
    def from_domain(cls, funcname, x_domain):
        xsize = int((x_domain[1] - x_domain[0]) * 10 + 1)
        xt = torch.linspace(x_domain[0], x_domain[1], xsize).unsqueeze(-1)
        return cls.from_function(funcname, xt)

    def __str__(self):
        xm = round(self.xt.min().item(), 2), round(self.xt.max().item(), 2)
        domainstr = f"[{xm[0]}:{xm[1]}]"
        return f"{self.funcname}_{domainstr}"

    def get_training_results(self, net, num_epoch):
        optimiser = self.optimfunc(net.parameters(), **self.optimkwargs)
        data_out = torch.zeros(num_epoch, self.xt.shape[0])
        loss_t = torch.zeros(num_epoch)

        og_loss = self.loss_func(net(self.xt), self.yt)
        loss = 0
        for epoch in range(num_epoch):
            optimiser.zero_grad()
            ypred = net(self.xt)

            loss = self.loss_func(ypred, self.yt)

            if DEBUG:
                print(epoch, loss)

            loss_t[epoch] = loss.item()
            data_out[epoch, :] = ypred.squeeze()

            loss.backward()
            optimiser.step()
        if DEBUG:
            print(f"First loss {og_loss} v final {loss}")
        return data_out.detach(), loss_t.detach()


def prep_animation(xd, yd, data, title, destfile):
    nepoch = data[0].shape[0]
    fig = plt.figure()
    ax = plt.axes()
    plt.title(title)
    ax.plot(xd, yd, ".")

    line_ref = []
    for i in range(len(data)):
        (liner,) = ax.plot([], [], lw=2)
        line_ref.append(liner)

    def init():
        for line in line_ref:
            line.set_data([], [])
        return line_ref

    def animate(i):
        for dnum, line in enumerate(line_ref):
            line.set_data(xd, data[dnum][i])
        return line_ref

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=nepoch, interval=20, blit=True, repeat=False
    )
    # i think interval is basically overwritten by fps
    # increase fps if you feel like the animation is moving too slowly
    if destfile:
        # secret skip
        anim.save(destfile, fps=30, extra_args=["-vcodec", "libx264"])
    plt.close()


@click.command()
@click.option("--hidden", "-h", type=int)
@click.option("--width", "-w", type=int)
@click.option("--activation", "-a", type=str, default="ReLU")
@click.option("-n", "--nepoch", type=int, default=200)
@click.option("--lr", "--learning-rate", type=float, default=0.002)
@click.option("--xdomain", type=str, default="-1:3")
@click.option("--func", "funcname", type=str, default="exp")
@click.option("--numtrains", type=int, default=50)
@click.option("--destdir", type=str, default="out_animations")
@click.option("--show/--no-show", default=True)
def main(hidden, width, activation, nepoch, lr, funcname, xdomain, numtrains, destdir, show):
    xdomain = [float(x) for x in xdomain.split(":")]

    # configure
    train = Trainer.from_domain(funcname, xdomain)
    # inherit/modify the get_tr_results for different training
    # pass in Trainer(name, xt, yt) for full control
    train.optimkwargs["lr"] = lr
    activation = get_activation(activation)  # get's relu by default
    # end config
    data_list = []
    print("Starting training")
    tr_start = pendulum.now()
    for i in range(numtrains):
        net = nn.Sequential(*make_net(hidden, width, activation))
        data, loss = train.get_training_results(net, nepoch)
        if np.any(np.isnan(loss.numpy())):
            raise RuntimeError(f"Nan loss found, drop lr. Currently lr={lr}")
        data_list.append(data.numpy())
    tm = pendulum.now() - tr_start
    print("Finished training in {}".format(tm.in_words()))

    destfile = os.path.join(destdir, f"{train}_{hidden}h{width}w_{activation}_{nepoch}e.mp4")
    print(f"Creating animation and saving to {destfile}")
    anim_start = pendulum.now()
    prep_animation(
        train.xt.detach(),
        train.yt.detach(),
        data_list,
        f"NN with {hidden} layers {width} wide and {activation} activation approximates {funcname}",
        destfile,
    )
    print("Finished animating in {}".format((pendulum.now() - anim_start).in_words()))
    if show:
        import webbrowser

        webbrowser.open_new_tab(os.path.abspath(destfile))


if __name__ == "__main__":
    main()
