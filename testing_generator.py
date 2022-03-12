"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


import os
import numpy as np
import math
import scipy.io as sio
import torch
import simulated_NV_DataSet
import NV_normalize_data as nrm

import flags as _


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

device = "cuda:0"


def testing_loader():
    folder_path = _.FLAGS['data_path'] + 'df_' + str(_.FLAGS['df']) + '/' + str(_.FLAGS['measures']) + '_measures/' + str(_.FLAGS['SIMULTANEOUS_FREQS']) + '_simultaneous_freqs/' + 'testing'


    # Measures tensors:
    sig_measurements_tensors = torch.ones([_.FLAGS['NUM_SIM_TEST_SAMPS'], 1, int(_.FLAGS['measures'])])
    
    # Target tensors:
    target_tensors = torch.ones([_.FLAGS['NUM_SIM_TEST_SAMPS'], 1, 8])
    # target_tensors = torch.ones([NUM_SIM_TEST_SAMPS, 1, 200]).to('cuda:0')
    # target_tensors = torch.ones([NUM_SIM_TEST_SAMPS, 1, 3]).to('cuda:0')
    
    normalization_params = []
    magnetic_fields = []
    cs_measures = []
    
    for smp in range(_.FLAGS['NUM_SIM_TEST_SAMPS']):
        data_struct = sio.loadmat(os.path.join(folder_path, 'test_CS_' + str(smp) + '_.mat'))['data_struct']
        
        try:
            curr_cs_measures = data_struct['curr_cs_per'][0][0]
            cs_measures.append(curr_cs_measures)
        except:
            curr_cs_measures = data_struct['cs_per'][0][0]
            cs_measures.append(curr_cs_measures)
        
        measures = torch.tensor(data_struct['measures'][0][0], requires_grad=False)
        measures = torch.transpose(measures, dim0=1, dim1=0)
        sig_measurements_tensors[smp, :, :] = nrm.norm_data(measures)

        curr_magnetic_vec = np.squeeze(np.array(data_struct['B_vec'][0][0]).T)
        magnetic_fields.append(math.sqrt(curr_magnetic_vec[0] ** 2 + curr_magnetic_vec[1] ** 2 + curr_magnetic_vec[2] ** 2))
        
        # target = torch.transpose(torch.tensor(data_struct['target'][0][0], requires_grad=False), dim1=0, dim0=1)
        target = torch.transpose(torch.tensor(data_struct['peak_locs'][0][0], requires_grad=False), dim1=0, dim0=1).to(device)
        # target = torch.tensor(data_struct['B_vec'][0][0], requires_grad=False)
        
        
        norm_tar, mean_, std_ = nrm.norm_targets(target)
        target_tensors[smp, :, :] = norm_tar
        normalization_params.append((target.to(device), mean_.to(device), std_.to(device)))

    # batch_size = 1
    test_params = {'batch_size': _.FLAGS['tst_batch_size'], 'shuffle': False}

    sig_meas_set = simulated_NV_DataSet.simulated_NV_DataSet(sig_measurements_tensors, target_tensors)
    test_gen = torch.utils.data.DataLoader(sig_meas_set, **test_params)
 
    return test_gen, normalization_params, magnetic_fields, np.squeeze(np.array(cs_measures))
