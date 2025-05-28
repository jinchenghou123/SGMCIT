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



def plot_loss(train_loss, eval_loss):
    len_loss = len(train_loss)

    fig = plt.figure(figsize = (3,2))
    plt.plot(list(range(len_loss)), train_loss)
    plt.plot(list(range(len_loss)), eval_loss)
    plt.legend(['train','test'])
    plt.show()
    
def plot_score(train_data, test_data, mesh, scores):
    fig = plt.figure(figsize = (5,5))
    plt.gca().set_aspect('equal')
    plt.scatter(train_data[:,0], train_data[:,1], s = 0.2, c = '#00b894')
    plt.scatter(test_data[:,0], test_data[:,1], s = 0.2, c = '#0984e3')
    colors = (scores[:, 0]**2+scores[:, 1]**2)**0.5
    im = plt.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1],colors, width=0.003,\
               cmap='Reds',scale_units='xy',angles='xy')
    plt.show()
