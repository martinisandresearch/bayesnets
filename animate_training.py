#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os

import click
import pendulum
import numpy as np

import swarm
from swarm import networks
from swarm import animator
from swarm import core


@click.command()
@click.option("--hidden", "-h", type=int, default=2)
@click.option("--width", "-w", type=int, default=2)
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
    train = core.Trainer(func, name, xdomain)
    # inherit/modify the get_tr_results for different training
    # pass in Trainer(name, xt, yt) for full control
    train.optimkwargs["lr"] = lr
    activationfunc = swarm.get_activation(activation)  # get's relu by default
    # end config
    data_list = []
    print("Starting training")
    tr_start = pendulum.now()
    for i in range(numtrains):
        net = networks.flat_net(hidden, width, activationfunc)
        data, loss = train.get_training_results(net, nepoch)
        if np.any(np.isnan(loss.numpy())):
            raise RuntimeError(f"Nan loss found, drop lr. Currently lr={lr}")
        data_list.append(data.numpy())
    tm = pendulum.now() - tr_start
    print("Finished training in {}".format(tm.in_words()))

    destfile = os.path.join(destdir, f"{train}_{hidden}h{width}w_{activation}_{nepoch}e.mp4")
    print(f"Creating animation and saving to {destfile}")
    anim_start = pendulum.now()
    animator.make_animation(
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
