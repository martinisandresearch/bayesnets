#  -*- coding: utf-8 -*-


import torch
from torch.nn import *


class xtanh(Module):
    def forward(self, x: torch.Tensor):
        return x - torch.tanh(x)
