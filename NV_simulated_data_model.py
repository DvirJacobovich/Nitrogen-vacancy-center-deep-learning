"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2023

"""

import torch
from torch import nn
import ProjPath
import flags as _


import NV_lorentzian_transformer as nv

from torch.nn import functional as F

def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)



class NV_simulated_data_model(nn.Module):
    def __init__(self):
        super(NV_simulated_data_model, self).__init__()
        self.proj_path = ProjPath.ProjPath().to(_.FLAGS['device'])  # .proj_seq
        # self.proj_path.apply(ProjPath.weights_init)
        self.lorentzian_transformer = nv.NV_lorentzian_transformer().to(_.FLAGS['device'])  
        self.dropout = nn.Dropout()
       
        self.decoder = nn.Sequential(
            nn.Dropout(_.FLAGS['FIRST_DROP']),
            
            nn.BatchNorm1d(1),
            nn.Linear(200, 4096),  # size: 4096
            nn.BatchNorm1d(1),

            nn.Conv1d(1, 32, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 2048
            nn.PReLU(num_parameters=32, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(32),
            

            nn.Dropout(_.FLAGS['DROPOUT_PROB']),

            nn.Conv1d(32, 64, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 1024
            nn.PReLU(num_parameters=64, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=64, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(64),

            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),

            nn.Conv1d(64, 128, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 512
            nn.PReLU(num_parameters=128, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=128, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(128),


            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(128, 256, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 256
            nn.PReLU(num_parameters=256, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=256, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(256),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(256, 512, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 128
            nn.PReLU(num_parameters=512, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(512),
            
            nn.Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=512, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(512),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(512, 1024, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 64
            nn.PReLU(num_parameters=1024, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(1024, 1024, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=1024, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(1024),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(1024, 2048, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 32
            nn.PReLU(num_parameters=2048, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(2048),
            
            nn.Conv1d(2048, 2048, kernel_size=(3,), stride=(1,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),
            nn.PReLU(num_parameters=2048, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(2048),
    
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(2048, 4096, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 16
            nn.PReLU(num_parameters=4096, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(4096),
          
            nn.Conv1d(4096, 4096, kernel_size=(4,), stride=(2,), padding=1, padding_mode=_.FLAGS['PADDING_MODE']),  # size: 8
            nn.PReLU(num_parameters=4096, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(4096),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(4096, 1024, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=1024, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(1024),
            
            nn.Conv1d(1024, 512, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=512, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(512),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(512, 128, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=128, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 32, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=32, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(32),
            
            
            nn.Dropout(_.FLAGS['DROPOUT_PROB']),
            
            nn.Conv1d(32, 16, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=16, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 1, kernel_size=(1,), stride=(1,), padding=0, padding_mode=_.FLAGS['PADDING_MODE']),  
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
            nn.BatchNorm1d(1),
            
            nn.Linear(8, 8),
          

        )

        self.linear = nn.Linear(int(_.FLAGS['measures']), 200)
        self.blinear = nn.Bilinear(1200, 200, 200) 

    def forward(self, proj_input):
        """
        :param proj_input: size: batch_size x 1 x 100
        :param mat_input: batch_size x 1 x 100 x 200
        :param magnetic_params_input: batch_size x 100 x 3
        :return: out - size: target size.
        """
        proj_out = self.proj_path(proj_input)  # proj_out size: batch_size x (1 x 1500)
        x1 = proj_out.view(proj_out.size(0), -1)  # size: batch_size x 1500
        x2 = x1[:, None, :] # size: batch_size x 1 x 1500
        
        x_mid = self.linear(proj_input) # size: batch_size x 1 x 200
        x3 = self.lorentzian_transformer(x_mid) # size: batch_size x 1 x 200

        x4 = self.blinear(x2, x3) 
        # x4 = self.decoder(self.linear2(x3))
        x5 = self.decoder(x4)
        return x5


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)  # std = 0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)  # std = 0.02
        nn.init.constant_(m.bias.data, 0)

