#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import os

import click
import pendulum

import torch

import swarm
import swarm.core
from swarm import networks, animator, core, regimes


def get_function(name):
    # let it error if it fails
    return getattr(torch, name)


@click.command()
@click.option("--hidden", "-h", type=int, default=2)
@click.option("--width", "-w", type=int, default=2)
@click.option("--activation", "-a", type=str, default="ReLU")
@click.option("-n", "--nepoch", type=int, default=200)
@click.option("--lr", "--learning-rate", type=float, default=0.002)
@click.option("--xdomain", type=str, default="-1:3")
@click.option("--func", "funcname", type=str, default="exp")
@click.option("--swarmsize", type=int, default=50)
@click.option("--destdir", type=str, default="out_animations")
@click.option("--show/--no-show", default=True)
def main(hidden, width, activation, nepoch, lr, funcname, xdomain, swarmsize, destdir, show):
    print(hidden, width, activation, nepoch, lr, funcname, xdomain, swarmsize, destdir, show)

    xdomain = [float(x) for x in xdomain.split(":")]
    xt = torch.linspace(xdomain[0], xdomain[1], 101)
    yt = get_function(funcname)(xt)
    afunc = swarm.core.get_activation(activation)

    trainer = regimes.SwarmTrainerBase(
        xt,
        yt,
        net_factory=lambda: networks.flat_net(hidden, width, afunc),
        num_epochs=nepoch,
        optimiser=lambda netp: torch.optim.SGD(netp, lr=lr, momentum=0.9),
    )

    runner = core.SwarmRunner.from_string("ypred,loss")

    tr_start = pendulum.now()
    results = runner.swarm_train(swarmsize, trainer.train_bee)
    tm = pendulum.now() - tr_start
    print("Finished swarm training in {}".format(tm.in_words()))

    xdstr = f"[{xdomain[0]}:{xdomain[1]}]"
    fname = f"{funcname}_{xdstr}_{hidden}h{width}w_{activation}_{nepoch}e.mp4"

    destfile = os.path.join(destdir, fname)
    print(f"Creating animation and saving to {destfile}")
    anim_start = pendulum.now()
    animator.make_animation(
        xt.detach().numpy(),
        yt.detach().numpy(),
        results["ypred"],
        f"NN with {hidden} layers {width} wide and {activation} activation approximates {funcname}",
        destfile,
    )
    print("Finished animating in {}".format((pendulum.now() - anim_start).in_words()))
    if show:
        import webbrowser

        print("Opening in browser")
        webbrowser.open_new_tab(os.path.abspath(destfile))


if __name__ == "__main__":
    main()
