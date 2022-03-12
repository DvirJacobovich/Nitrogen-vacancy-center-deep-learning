"""
Dvir Jacobovich NV deep leaning simulation 
John Howell lab 2022

"""


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import flags as _


def model_rec_plots(net_rec_batch, target_batch, smp_freqs, test_type):
    # net_rec_batch1 = torch.from_numpy(net_rec_batch[0, :, :].detach().numpy())
    # target_batch1 = torch.from_numpy(target_batch[0, :, :].detach().numpy())
    rand_samp = torch.randint(net_rec_batch.size()[0], (1,))
    a, b = net_rec_batch, target_batch
    plt.plot(smp_freqs, (a[rand_samp.item(), 0, :].cpu().detach().numpy()), label='Model rec', color='r')
    plt.plot(smp_freqs, b[rand_samp.item(), 0, :].cpu().detach().numpy(), label='Raster scan', color='b')

    # plt.plot(torch.squeeze(plt.plot(target_batch[0, 0, :])), label='Target', color='b')
    # plt.title('Target Lorentians')
    title = 'Net Recs and Target Lorentzians ' + test_type
    plt.title(title)
    plt.xlabel('MW Frequency [MHz]')
    plt.ylabel('Absorption normalized (Arb. Units)')
    plt.legend()
    plt.savefig('rec')
    # plt.show()


def plot_losses(train_err, valid_err, max_epochs):
    max_epochs = 10
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(range(len(train_err)), train_err, color='red', label='Training loss')
    ax1.legend(loc=0)
    # plt.legend()
    ax1.plot(range(len(valid_err)), valid_err, color='green', label='Validation loss')
    # ax1.set_xlim(left=1)
    ax1.set_ylim(0, 0.02)

    ax1.legend(loc=1)
    ax1.grid(axis='y', linestyle='-')

    plt.xticks(np.arange(len(train_err) * _.FLAGS['VERBOSE']), [str(i) for i in range(len(train_err) * _.FLAGS['VERBOSE'])])
    ax1.set_xlabel('Total batches Num')
    ax1.set_ylabel('MAE loss value [MHz]')

    min_valid = min(valid_err)
    min_label = 'Min validation err: ' + format(min_valid, ".6f")
    ax2.plot(range(1, max_epochs + 1), np.ones(max_epochs) * min_valid, linestyle='--', label='Min validation err: ' +
                                                                        format(min_valid, ".6f"))
    ax2.set_xlabel('Number of epochs')
    # ax2.cla()
    ax2.legend(loc=2)
    # ax2.set_xlim(left=2)
    
    ax2.grid(axis='x', linestyle='-')
    plt.title('Training and Validation MAE losses')
    name = 'Losses_plots' + str(min(valid_err)).replace('.', '_') + '.png'
    plt.savefig(name)

    # plt.show()



def NV_show_dif(net_rec_batch, target_batch):
    # net_rec_batch1 = torch.from_numpy(net_rec_batch[0, :, :].detach().numpy())
    # target_batch1 = torch.from_numpy(target_batch[0, :, :].detach().numpy())
    a, b = net_rec_batch, target_batch
    plt.plot((a[0, 0, :].detach().numpy()), label='Net rec', color='r')
    plt.title('net_rec')
    plt.show()

    plt.plot(b[0, 0, :].detach().numpy(), label='Target', color='b')
    plt.title('target')
    plt.show()

    # fig = plt.figure()
    # plt.imshow(net_rec)
    # plt.colorbar(label="Reconstruction Network output", orientation="vertical")
    # plt.title('Reconstruction Network output with 25%')
    #
    # plt.show()








def show_diff_nv(net_rec, original, check):
    fig = plt.figure(figsize=(10, 5), dpi=50)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title(check + ' check Net Rec (Normed) 25%')
    b1 = ax1.imshow(net_rec)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b1, cax=cax1, orientation='vertical')

    ax2 = fig.add_subplot(1, 2, 2)
    plt.title(check + ' check Orig (Normed)')
    b2 = ax2.imshow(original)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b2, cax=cax2, orientation='vertical')

    plt.show()
