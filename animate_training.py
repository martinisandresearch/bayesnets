#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os

import attr
import click
import torch
from torch import optim, nn

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

plt.rcParams["figure.figsize"] = (14.0, 7.0)
sns.set()


DEBUG = False


def make_net(hidden_depth, width):
    assert hidden_depth >= 1
    yield nn.Linear(1, width)
    yield nn.ReLU()
    for i in range(hidden_depth - 1):
        yield nn.Linear(width, width)
        yield nn.ReLU()
    yield nn.Linear(width, 1)


@attr.s
class Trainer:
    funcname = attr.ib(type=str)
    x_domain = attr.ib()
    loss_func = attr.ib(default=nn.MSELoss())
    optim_factory = attr.ib(default=lambda net: optim.SGD(net.parameters(), lr=0.002, momentum=0.9))

    xt = attr.ib(init=False)
    yt = attr.ib(init=False)

    def __attrs_post_init__(self):
        try:
            func = getattr(torch, self.funcname)
        except AttributeError as ex:
            raise AttributeError(f"Unable to find {self.funcname} in torch0") from ex
        xsize = int((self.x_domain[1] - self.x_domain[0]) * 10 + 1)
        self.xt = torch.linspace(self.x_domain[0], self.x_domain[1], xsize).unsqueeze(-1)
        self.yt = func(self.xt)

    def __str__(self):
        return f"{self.funcname}_[{self.x_domain[0]}:{self.x_domain[1]}]"

    def get_training_results(self, net, num_epoch):
        optimiser = self.optim_factory(net)
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

    # initialization function: plot the background of each frame
    def init():
        for line in line_ref:
            line.set_data([], [])
        return line_ref

    # animation function.  This is called sequentially
    def animate(i):
        #     print(i)
        for dnum, line in enumerate(line_ref):
            line.set_data(xd, data[dnum][i])
        return line_ref

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=nepoch, interval=20, blit=True
    )
    anim.save(destfile, fps=30, extra_args=["-vcodec", "libx264"])
    plt.close()


@click.command()
@click.option("--hidden", "-h", type=int)
@click.option("--width", "-w", type=int)
@click.option("-n", "--nepoch", type=int, default=200)
@click.option("--xdomain", type=str, default="-1:3")
@click.option("--func", "funcname", type=str, default="exp")
@click.option("--numtrains", type=int, default=50)
@click.option("--destdir", type=str, default="out_animations")
def main(hidden, width, funcname, nepoch, xdomain, numtrains, destdir):
    xdomain = [float(x) for x in xdomain.split(":")]
    train = Trainer(funcname, xdomain)

    data_list = []
    print("Starting training")
    for i in range(numtrains):
        net = nn.Sequential(*make_net(hidden, width))
        data, loss = train.get_training_results(net, nepoch)
        data_list.append(data.numpy())
    print("Finished training ")

    destfile = os.path.join(destdir, f"{train} {hidden}h{width}w {nepoch}epoch.mp4")
    print(f"Creating animation and saving to {destfile}")
    prep_animation(
        train.xt.detach(),
        train.yt.detach(),
        data_list,
        f"Approxmiated {funcname} with {hidden} Hidden of {width} Width",
        destfile,
    )


if __name__ == "__main__":
    main()
