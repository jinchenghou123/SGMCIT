import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class cscore_matching():
    
    def __init__(self, model, device, dx = 1, dz = 1, learning_rate = 1e-3, n_slices = 1):
        
        self.model = model
        self.device = device
        self.n_slices = n_slices
            
        
        
        self.dx = dx
        self.dz = dz
        
        # setup optimizer
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, \
                                              weight_decay=1e-5)

    
    def load_data(self, training_data, test_data, batch_size = 64):
        self.train_dataloader = DataLoader(training_data, batch_size=batch_size,\
                                           shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, \
                                          shuffle=True)
    
    def train(self, epoch, debug = 0):
        
        self.train_loss_all = []
        self.eval_loss_all = []
        for e in range(epoch):
            loss_epoch = []
            for it, train_x in enumerate(self.train_dataloader):
                x_z = train_x.to(self.device)
                x = x_z[:,:self.dx].reshape(-1, self.dx)
                z = x_z[:,self.dx:].reshape(-1, self.dz)
                loss = self.get_loss(x,z)

                # compute gradients
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_epoch.append(loss.item())
            
            loss_epoch_mean = torch.mean(torch.Tensor(loss_epoch))
                
            if  e % 5 == 0 and debug>0: 
                self.train_loss_all.append(loss_epoch_mean)
                self.model.eval()
                eval_loss_mean = self.evaluation()
                self.eval_loss_all.append(eval_loss_mean)
                self.model.train()
            
            if  e % 20 == 0 and debug>0: 
                print("epoch ",e, " train_loss_mean ", loss_epoch_mean, \
                      "eval_loss_mean ", eval_loss_mean)
                
        if debug>0:    
            print("end training!")
    
    def evaluation(self):
        # with torch.no_grad():
        loss_eval = []
        for it, test_x in enumerate(self.test_dataloader):
            x_z = test_x.to(self.device)
            x = x_z[:,:self.dx].view(-1, self.dx)
            z = x_z[:,self.dx:].view(-1, self.dz)
            loss = self.get_loss(x,z)
            loss_eval.append(loss.item())
            
        return torch.mean(torch.Tensor(loss_eval))
        
    def get_loss(self, x, z, v=None):
        """Compute loss

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor, optional): sampled noises. Defaults to None.

        Returns:
            loss
        """

        v = self.get_random_noise(x, self.n_slices)
        loss = self.condition_ssm_vr_loss(x, z, v) 
        

        return loss
    
    
    def condition_ssm_vr_loss(self, x, z, v):
        x = x.unsqueeze(0).expand(self.n_slices, *x.shape) # (n_slices, b, ...)
        x = x.contiguous().view(-1, *x.shape[2:]) # (n_slices*b, ...)
        x = x.requires_grad_()
        score = self.model.score(x, z) # (n_slices*b, ...)
        sv = torch.sum(score * v) # ()
        loss1 = torch.norm(score, dim=-1) ** 2 * 0.5 # (n_slices*b,)
        gsv = torch.autograd.grad(sv, x, create_graph=True)[0] # (n_slices*b, ...)
        loss2 = torch.sum(v*gsv, dim=-1) # (n_slices*b,)
        loss = (loss1 + loss2).mean() # ()
        return loss

    def get_random_noise(self, x, n_slices=None):
        """Sampling random noises

        Args:
            x (torch.Tensor): input samples
            n_slices (int, optional): number of slices. Defaults to None.

        Returns:
            torch.Tensor: sampled noises
        """
        if n_slices is None:
            v = torch.randn_like(x, device=self.device)
        else:
            v = torch.randn((n_slices,)+x.shape, dtype=x.dtype, device=self.device)
            v = v.view(-1, *v.shape[2:]) # (n_slices*b, 2)

        
        return v


        
    

    
    
    