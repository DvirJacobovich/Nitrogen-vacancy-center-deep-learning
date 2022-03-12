"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""

import torch

use_cuda = torch.cuda.is_available()

FLAGS = {}
# Define Parameters
FLAGS['data_path'] = 'matlab_code/COmpSENS/' 
FLAGS['device'] = torch.device("cuda:0" if use_cuda else "cpu")

# Compressed sensing parameters:
FLAGS['measures'] = 50
FLAGS['df'] = 2
FLAGS['SIMULTANEOUS_FREQS'] = 3


# Data samples parameters:
FLAGS['NUM_SIM_SAMPS'] = 4000
FLAGS['NUM_VALID_SAMPS'] = 120
FLAGS['NUM_SIM_TEST_SAMPS'] = 120
FLAGS['epochs'] = 400 
FLAGS['batch_size'] = 128 
FLAGS['valid_batch_size'] = 120  # always equal to NUM_VALID_SAMPS to max prediction
FLAGS['tst_batch_size'] = 1  # always equal to 1.
FLAGS['shuffle'] = True
FLAGS['VERBOSE'] = 3

FLAGS['breaking_points'] = [150, 200, 250, 300, 350]


# Learning process parameters:
FLAGS['lr'] = 1e-3 / 6  # learning rate
FLAGS['beta1'] = 0.9  # momentum for ADAM
FLAGS['beta2'] = 0.999
FLAGS['weight_decay'] = 0

# Clipping grad options:
FLAGS['clipping_grad'] = ['none']  # options: ['none', 'unscale', 'clip_val_norm', 'register_hook', 'grad_penalty']
FLAGS['clipping_value'] = 1  # 0.5  # Clip_value (float or int): maximum allowed value of the gradients
FLAGS['max_clip_norm'] = 1

# Model hyper-parameters
FLAGS['PADDING_MODE'] = 'reflect'  # options: [reflect, replicate, zeros, circular]'
FLAGS['DROPOUT_PROB'] = 0.5
FLAGS['FIRST_DROP'] = 0.2

# NV Lorentzian transformer parameters:
FLAGS['_ntokens'] = 200  # was 200. Size of vocabulary. In our case we have 200 ("words") output size.
FLAGS['_emsize'] = 200  # was 200. Embedding dimension
FLAGS['_d_hid'] = 8  # was 200 or 1. Dimension of the feedforward network model in nn.TransformerEncoder
FLAGS['_nlayers'] = 1  # was 2. Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
FLAGS['_nhead'] = 2  # was 20. Number of heads in nn.MultiheadAttention
FLAGS['_dropout'] = 0.5  # was 0.2 Dropout probability
