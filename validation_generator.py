"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""

import os
import scipy.io as sio
import torch
import simulated_NV_DataSet
import NV_normalize_data as nrm
from sklearn.preprocessing import StandardScaler

import flags as _


def valid_gen():
    folder_path = _.FLAGS['data_path'] + 'df_' + str(_.FLAGS['df']) + '/' + str(_.FLAGS['measures']) + '_measures/' + str(_.FLAGS['SIMULTANEOUS_FREQS']) + '_simultaneous_freqs/' + 'validation'
    
    # Measures tensors:
    # sig_measurements_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 1, int((cs_per/100) * freq_samps)]).to('cuda:0')
    sig_measurements_tensors = torch.ones([_.FLAGS['NUM_VALID_SAMPS'], 1, int(_.FLAGS['measures'])])
    
    # Target tensors:
    # target_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 1, 8]).to('cuda:0')
    target_tensors = torch.ones([_.FLAGS['NUM_VALID_SAMPS'], 1, 8])
    
    for smp in range(_.FLAGS['NUM_VALID_SAMPS']):
        data_struct = sio.loadmat(os.path.join(folder_path, 'valid_CS_' + str(smp) + '_.mat'))['data_struct']
        
        measures = torch.tensor(data_struct['measures'][0][0], requires_grad=False)  # DOTO: req_grad changed to False.
        measures = torch.transpose(measures, dim0=1, dim1=0)
        sig_measurements_tensors[smp, :, :] = nrm.norm_data(measures)


        
        # target = torch.transpose(torch.tensor(data_struct['target'][0][0], requires_grad=False), dim1=0, dim0=1)
        target = torch.transpose(torch.tensor(data_struct['peak_locs'][0][0], requires_grad=False), dim1=0, dim0=1)
        # target = torch.tensor(data_struct['B_vec'][0][0], requires_grad=False)
        
        target_tensors[smp, :, :] = nrm.norm_targets(target)[0]
        
      
    valid_params = {'batch_size': _.FLAGS['valid_batch_size'], 'shuffle': False}  # No point of shuffeling validation
    sig_meas_set = simulated_NV_DataSet.simulated_NV_DataSet(sig_measurements_tensors, target_tensors)
    valid_gen1 = torch.utils.data.DataLoader(sig_meas_set, **valid_params)

    return valid_gen1
