"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


import torch
from torch import nn
import flags as _


class ProjPath(nn.Module):
    def __init__(self):
        super(ProjPath, self).__init__()
        self.proj_seq = nn.Sequential(
            nn.Linear(int(_.FLAGS['measures']), 1200),
            nn.BatchNorm1d(1),
          
        )

    def forward(self, Input):
        return self.proj_seq(Input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        nn.init.constant_(m.bias.data, 0)
