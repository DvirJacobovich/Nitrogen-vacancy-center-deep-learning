"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


import torch
import NV_simulated_data_model as nv_model
import numpy as np
import testing_generator
import NV_plots
import calculate_MHz_point_ratio
from torch.nn import functional as F
from matplotlib import pyplot as plt
import os

import flags as _


print('CS_MEASURES: ', _.FLAGS['measures'], ',', 'SIMULTANEOUS_FREQ: ', _.FLAGS['SIMULTANEOUS_FREQS'], ',','\n')

# full_path = '350_epochs0_0001'
full_path = '4008_347159018740058e-05'

model = nv_model.NV_simulated_data_model().to(_.FLAGS['device'])
    
model.load_state_dict(torch.load(full_path))
_measure, normalization_params, magnetic_fields, cs_measures = testing_generator.testing_loader()

model.eval()

all_err = []

for batch_num, (sig_meas_batch, target_batch) in enumerate(_measure):
    curr_origin_data_batch, curr_mean, curr_std = normalization_params[batch_num]
    sig_meas_batch = sig_meas_batch.to(_.FLAGS['device'])
    curr_origin_data_batch = curr_origin_data_batch.to(_.FLAGS['device'])
    
    model_output = model(sig_meas_batch)
    model_output_non_normalized = (model_output * curr_std) + curr_mean
    
    print('Magnetic Field: ', magnetic_fields[batch_num], '\n')
    tst_err = F.l1_loss(torch.squeeze(curr_origin_data_batch), torch.squeeze(model_output_non_normalized))
    print('Original non normalized target: ')
    x = curr_origin_data_batch.detach().cpu().numpy()[0].tolist()
    print([float("{0:.4f}".format(i)) for i in x], '[MHz]\n')

    print('Model output non normalized: ')
    # print(model_output_non_normalized.detach().cpu().numpy()[0][0], '\n')
    x2 = model_output_non_normalized.detach().cpu().numpy()[0][0].tolist()
    print([float("{0:.4f}".format(i)) for i in x2], '[MHz]\n')
    
    all_err.append(float(tst_err.item()))
    print('Mean Absolut Error: ', tst_err.item(), '[MHz]\n')
    print('\n')

average_err = torch.mean(torch.tensor(all_err)).item()
print('Average Error: ', torch.mean(torch.tensor(all_err)).item(), '[MHz]')

############################################################################################

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111)

xs, ys = zip(*sorted(zip(magnetic_fields, all_err)))
ax1.plot(xs, ys)

ax1.grid(axis='y', linestyle='-')
ax1.grid(axis='x', linestyle='-')
plt.title("Magnetic field magnitude Vs. MAE")
plt.xlabel('Magnetic Field [Gaus]')
plt.ylabel('Mean Absolut Error [MHz]')
plt.savefig('mag_field')


############################################################################################

fig = plt.figure(figsize=(7, 7))
ax2 = fig.add_subplot(111)

xs2, ys2 = zip(*sorted(zip(cs_measures, all_err)))
ax2.plot(xs2, ys2)

ax2.grid(axis='y', linestyle='-')
ax2.grid(axis='x', linestyle='-')
plt.title("N. CS Measures Vs. MAE")
plt.xlabel('CS Measures percentage')
plt.ylabel('Mean Absolut Error [MHz]')
plt.savefig('cs_measures')

#############################################################################################

fig = plt.figure(figsize=(7, 7))
ax3 = fig.add_subplot(111)

xs3, ys3 = zip(*sorted(zip(magnetic_fields, cs_measures)))
ax3.plot(xs3, ys3)

ax3.grid(axis='y', linestyle='-')
ax3.grid(axis='x', linestyle='-')
plt.title("Magnetic field magnitude Vs. Percentage")
plt.xlabel('Magnetic Field [Gauss]')
plt.ylabel('Percentage')
plt.savefig('MagVsPer')


############################################################################################

magnetics = xs
measure_pers = xs2
errs = ys

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.plot(measure_pers, errs, color='red', label='Mean Absolute Error [MHz]')
ax1.legend(loc=1)

ax1.grid(axis='x', linestyle='-')

# plt.xticks(np.arange(len(errs)), np.arange(len(errs)))
ax1.set_xlabel('CS Percentage (50 / full window size)', fontsize=11)
ax1.set_ylabel('Mean Absolute Error [MHz]', fontsize=11)

ax2.plot(measure_pers, np.flip(np.array(magnetics)), color='blue', label='Magnetic Field [Gauss]')
ax2.set_ylabel('Magnetic Fields [Gauss]', fontsize=11)
ax2.legend(loc=2)


# ax3 = ax2.twinx()
ax1.plot(measure_pers, np.ones(len(measure_pers)) * average_err, color='green', linestyle='--', label='Averaged error: ' +
                                                                        format(average_err, ".6f") + ' [MHz]')
ax1.legend(loc=3)


ax2.grid(axis='y', linestyle='-')

plt.title('Fixed Measurements number: 50',fontsize=12)
plt.suptitle('Magnetic Field & MAE Vs. Compressed Sensing Ratio',fontsize=14, y=0.94)
# plt.suptitle(r'\textbf{Magnetic Field & MAE Vs. Compressed sensing ratio}',fontsize=14, y=0.94)

name = 'total' + '.png'
plt.savefig(name)






# magnetics = xs
# measure_pers = xs2
# errs = ys

# fig = plt.figure(figsize=(8, 8))
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()

# ax1.plot(measure_pers, errs, color='red', label='Mean Absolute Error')
# # ax1.legend(loc=1)

# ax1.grid(axis='x', linestyle='-')

# # plt.xticks(np.arange(len(errs)), np.arange(len(errs)))
# ax1.set_xlabel('CS Percentage', fontsize=11)
# ax1.set_ylabel('Mean Absolute Error [MHz]', fontsize=11)

# ax2.plot(measure_pers, np.flip(np.array(magnetics)), color='blue', label='Magnetic Field')
# ax2.legend(loc=2)
# ax2.set_ylabel('Magnetic Fields [Gaus]', fontsize=11)

# ax1.plot(measure_pers, np.ones(len(measure_pers)) * average_err, color='green', linestyle='--', label='Averaged error: ' +
#                                                                         format(average_err, ".6f"))
# ax1.legend(loc=1)

# ax2.grid(axis='y', linestyle='-')

# plt.title('Fixed Measurements number: 50',fontsize=12)
# plt.suptitle('Magnetic Field & MAE Vs. Compressed Sensing Ratio',fontsize=14, y=0.94)
# # plt.suptitle(r'\textbf{Magnetic Field & MAE Vs. Compressed sensing ratio}',fontsize=14, y=0.94)

# name = 'total' + '.png'
# plt.savefig(name)