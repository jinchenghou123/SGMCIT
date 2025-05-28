import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn

def condition_langevin_dynamics(
    score_fn,
    x, z,
    eps=0.1,
    n_steps=1000
):
    """Langevin dynamics

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        x (torch.Tensor): input samples
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    for i in range(n_steps):
        x = x + eps/2. * score_fn(x, z).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x