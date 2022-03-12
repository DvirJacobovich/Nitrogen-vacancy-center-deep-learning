"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

This project simulates the reconstruction of 8 NV peak locations with different magnetic field magnitude,
and can be set to different lineshape samples as well as different CS ratio and different number of 
simultaneous MW frequencies.
"""

import os
import time
import math
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import NV_plots
import NV_simulated_data_model
import validation_generator as vlg
import training_generator as trg
from torch.cuda.amp import GradScaler

import flags as _

torch.backends.cudnn.benchmark = True

print('device: ', _.FLAGS['device'], '\n')

print("CS MEASURES: ", _.FLAGS['measures'], ",", "  SIMULTANEOUS_FREQS:", _.FLAGS['SIMULTANEOUS_FREQS'], ",", "df: ", _.FLAGS['df'], ',\n')

print('TRAINING SET SIZE: ', _.FLAGS['NUM_SIM_SAMPS'], ",", " epochs: ", _.FLAGS['epochs'], ",", "  batch_size: ", _.FLAGS['batch_size'], ",  initial lr: ", _.FLAGS['lr'], ',\n')

print('VALIDATION SET SIZE: ', _.FLAGS['NUM_VALID_SAMPS'], ",", 'validation batch size: ', _.FLAGS['valid_batch_size'], ',\n')
print('verbose: ', _.FLAGS['VERBOSE'], '\n')

_tr_measure_gen = trg.tr_gen()
_vd_measure_gen = vlg.valid_gen()

print('Finished loading data into Pytorch tensors.\n')

MAE_criterion = nn.L1Loss()
training_losses, validation_losses = [], []


def train_NV_model(model: NV_simulated_data_model, criterion=nn.L1Loss(), optimization_loss='TV'):
    """
    Train algorithm using MAE loss criterion and smoothed version of L1 norm
    for a sparse output representation, or TV minimization for the original
    network output representation.
    """
    min_valid_model = copy.deepcopy(model)
    # optimizer = optim.Adam(model.parameters(), lr=_.FLAGS['lr'], betas=(_.FLAGS['beta1'], _.FLAGS['beta2']))  
    optimizer = optim.RAdam(model.parameters(), lr=_.FLAGS['lr'], betas=(_.FLAGS['beta1'], _.FLAGS['beta2']), eps=1e-08, weight_decay=1e-4)
    for g in optimizer.param_groups:
        print('Learning rate expected: ', _.FLAGS['lr'], ' Actual Learning rate: ', g['lr'], '\n')
    
    # Loop over epochs
    for epoch in range(1, _.FLAGS['epochs'] + 1):
        # Training
        for batch_num, (sig_meas_batch, target_batch) in enumerate(_tr_measure_gen):
            optimizer.zero_grad()

            # Transfer to device (cpu, GPU cuda:0, xla)
            sig_meas_batch = sig_meas_batch.to(_.FLAGS['device'])
            target_batch = target_batch.to(_.FLAGS['device'])
            
            # Feed farword operation
            net_rec = model(sig_meas_batch)
            primal_loss = criterion(net_rec, target_batch)

            loss_update(batch_num, epoch, primal_loss, net_rec, target_batch)
            
            if batch_num % _.FLAGS['VERBOSE'] == 0 and batch_num != 0:
                training_losses.append(primal_loss.item())
            
                # Validation test
                min_valid_model = validation_test(model, min_valid_model, epoch, validation=True)
                model.train()
                
            # Performing back-propogation and avoiding Gradient exploding options:
            gradient_clipping(primal_loss, model, optimizer)
            
        
        if epoch in _.FLAGS['breaking_points']:
            NV_plots.plot_losses(training_losses, validation_losses, _.FLAGS['epochs'])
            min_valid_name = str(epoch) + '_epochs' + str(format(min(validation_losses), ".4f")).replace('.', '_') 
            torch.save(min_valid_model.state_dict(), min_valid_name)
            
    min_valid_name = str(_.FLAGS['epochs']) + str(min(validation_losses)).replace('.', '_')
    torch.save(min_valid_model.state_dict(), min_valid_name)
    NV_plots.plot_losses(training_losses, validation_losses, _.FLAGS['epochs'])

        
    

def validation_test(model, min_valid_model, epoch, validation):
    """
    validation test
    """
    if validation:
        with torch.set_grad_enabled(False):
            model.eval()
            valid_criterion = nn.L1Loss()
            for i, (sig_meas_valid, target_valid) in enumerate(_vd_measure_gen):
                sig_meas_valid = sig_meas_valid.to(_.FLAGS['device'])
                target_valid = target_valid.to(_.FLAGS['device'])
                
                net_rec = model(sig_meas_valid)
                curr_valid_err = valid_criterion(net_rec, target_valid)

                validation_losses.append(curr_valid_err.item())
                print('Validation loss: ', validation_losses[-1], 'current min valid value: ', min(validation_losses))
    
                if curr_valid_err.item() == min(validation_losses):
                    min_valid_model = copy.deepcopy(model)
                    min_valid = curr_valid_err.item()
                    print('Entered min valid scope, new min valid is now: ', min_valid, '\n\n')
                
                else:
                    print('\n')
                
    return min_valid_model



def gradient_clipping(err, net, optimizer):
    """
    Executing gradient clipping to avoid gradient exploding, and getting a
    more stable learning process.
    # All options = ['nothing', 'unscale', 'clip_val_norm', 'register_hook', 'grad_penalty']
    """
    clipping_value = 1  # 0.5  # Clip_value (float or int): maximum allowed value of the gradients
    max_clip_norm = 1

    if 'none' in _.FLAGS['clipping_grad']:
        err.backward(retain_graph=True)  # perform back-propagation
        optimizer.step()  # update the weights.

    else:
        # option 1:
        if 'unscale' in _.FLAGS['clipping_grad']:
            scaler = GradScaler()  # TODO put it outside scope if used.
            scaler.scale(err).backward(retain_graph=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=_.FLAGS['max_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

        else:
            # Option 2:
            if 'grad_penalty' in _.FLAGS['clipping_grad']:
                grad_norm = gradient_penalty(err, net)
                err = err + grad_norm
                print('grad_norm_penalty: ', grad_norm, '->', 1 * grad_norm, '\n')

            err.backward(retain_graph=True)  # perform back-propagation

            # Option 3:
            if 'clip_val_norm' in _.FLAGS['clipping_grad']:
                torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=_.FLAGS['max_clip_norm'])
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=_.FLAGS['clipping_value'])

            # Option 4:
            if 'register_hook' in _.FLAGS['clipping_grad']:
                for p in net.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -_.FLAGS['clipping_value'], _.FLAGS['clipping_value']))

            optimizer.step()
            
    return err




def loss_update(i, epoch, primal_loss, net_rec, target_batch):
    print('Epoch num %d out of %d, Batch num %d out of %d:' %
          (epoch, _.FLAGS['epochs'], i + 1, math.ceil(int(_.FLAGS['NUM_SIM_SAMPS'] / _.FLAGS['batch_size']) + 1)))

    print('MAE training loss: ', primal_loss.item(), '\n')
  

def gradient_penalty(loss, model):
    """
    Returns loss's gradient_penalty as an option to avoid grad exploding by
    passing the gradients norms.
    :return: The gradient's l2 norm
    """
    # Creates gradients
    grad_params = torch.autograd.grad(outputs=loss,
                                      inputs=model.parameters(),
                                      create_graph=True, allow_unused=True)

    # Computes the penalty term and adds it to the loss
    grad_norm = 0
    # grads_lst = []
    for grad in grad_params:
        if grad is None:
            continue
        # grads_lst.append(grad)
        grad_norm += grad.pow(2).sum()

    grad_norm = math.sqrt(grad_norm)
    return grad_norm



if __name__ == '__main__':
    model = NV_simulated_data_model.NV_simulated_data_model().to(_.FLAGS['device'])
    # model.apply(NV_simulated_data_model.weights_init)
    start_time = time.time()
    train_NV_model(model, optimization_loss='sparsity')
    print('time: ', (time.time() - start_time) / 3600, 'hours')
    print('min_valid: ', min(validation_losses))

