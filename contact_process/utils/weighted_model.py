from torch import nn
import torch
from math import exp
import numpy as np
import matplotlib as mpl
import os
import sys
import matplotlib.pyplot as plt
import tikzplotlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchdiffeq import odeint_adjoint as odeint
from matplotlib.ticker import FuncFormatter

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    def netto(self,y):
        return self.net(y)

class AvgsODEFunc(nn.Module):
    """
        A model that is a weighted average of other model
    """
    def __init__(self,n_train,weights):
        super(AvgsODEFunc, self).__init__()
        if len(weights) != n_train:
                raise ValueError("weights must have length n_train")
        self.funcs = nn.ModuleList([ODEFunc() for _ in range(n_train)])
        self.weights = weights
    def forward(self, t, y):
        f_ys = torch.tensor([w*f(t,y) for w,f in zip(self.weights,self.funcs)])
        return torch.sum(f_ys)/self.weights.sum() 
    def netto(self,y):
        f = torch.zeros_like(y)
        for idx in range(len( self.weights )):
                f = f+self.weights[idx] * self.funcs[idx].netto(y)
        return f/self.weights.sum() 
