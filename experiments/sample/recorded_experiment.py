#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

from swarm import core, io, regimes
import torch


def main():
    """WIP - ignore for now"""
    x = torch.linspace(-6, 6, 101)
    y = torch.sin(x)

    metadata = {
        "name": "recorded_sin",
        "domain": (-6, 6),
        "seed": 10,
    }
