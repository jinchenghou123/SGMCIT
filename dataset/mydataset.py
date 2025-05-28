import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
# import tensorflow as tf

from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import math
import logging
import functools
from torch.utils.data import Dataset


class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        return sample