#  -*- coding: utf-8 -*-


import torch
from torch.nn import ReLU, Tanh


class xTanH(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x - torch.tanh(x)

class TanH(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.tanh(x)