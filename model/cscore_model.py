import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class Swish(nn.Module):
    def __init__(self, dim=-1):
        """Swish activ bootleg from
        https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299

        Args:
            dim (int, optional): input/output dimension. Defaults to -1.
        """
        super().__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class cMLP(nn.Module):
    def __init__(
        self, 
        input_dim=2,
        output_dim=2,
        units=[300, 300]
    ):
        """Toy MLP from
        https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198

        Args:
            input_dim (int, optional): input dimensions. Defaults to 2.
            output_dim (int, optional): output dimensions. Defaults to 1.
            units (list, optional): hidden units. Defaults to [300, 300].
            swish (bool, optional): use swish as activation function. Set False to use
                soft plus instead. Defaults to True.
            dropout (bool, optional): use dropout layers. Defaults to False.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                Swish(out_dim),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, z, sigmas=None):
        x_z = torch.cat((x,z),1)
        return self.net(x_z)
    
class cScore(nn.Module):
    def __init__(self, net):
        """
        A simple score model
        """
        super().__init__()
        self.net = net

    def forward(self, x, z):
        return self.net(x, z)

    def score(self, x, z, sigma=None):
        if sigma is None:
            score_x = self.net(x, z)
        else:
            score_x = self.net(x, z, sigma)
        return score_x

