#IMPORTS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import sys
import torch

# SETTINGS


sys.path.append('../../')
from lib.datasets import * 
from lib.utils import *
from lib.vis import *
from lib.vae.models import *
import sci.lib.vae.parser as wparse



def plotNODE(predNODE, labelPOD, lossTrain, lossVal, itr, train_win, res_folder, args):
    """Plot node predictions"""
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    fig.suptitle('Reconstruction of {} POD temporal modes - Epoch: {}\n LDIM {}, LENC {}, UENC {}, LNODE {}, LDEC {}, UDEC {}, LR {}'.format(args.dataset,itr,
            args.latent_dim, args.layers_enc, args.units_enc, args.layers_node,
                args.layers_dec, args.units_dec, args.lr), fontsize=24)

    filename = str('%04d' % itr)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.45, wspace=0.25)

    t_steps = np.linspace(0, 100, labelPOD.shape[0])

    for k in range(8):

        ax = fig.add_subplot(5, 2, k + 1)
        ax.plot(t_steps, labelPOD[:, k], color='r', linewidth=2.5, alpha=1, label='POD')
        ax.plot(t_steps, predNODE[0, :, k], 'k--', linewidth=2.5, label=args.model)
        ax.axvline(x=t_steps[train_win - 1], color='k')

        ax.set_ylabel('$a_{%d}$' % (k + 1), rotation=0, size=25, labelpad=10)

        if k == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=25)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

    ax = fig.add_subplot(5, 2, 10)
    ax.plot(lossTrain, '-k', linewidth=2.0, label='Loss Train')
    ax.plot(lossVal, '--r', linewidth=2.0, label='Loss Validation')

    plt.xlabel('Epoch', fontsize=24)
    legend = ax.legend(loc=0, ncol=1, prop={'size': 20}, bbox_to_anchor=(0, 0, 1, 1), fancybox=True, shadow=False)
    plt.setp(legend.get_title(), fontsize='large')
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    plt.savefig(args.out_dir+'/'+args.model+"/modes.pdf", format="pdf", bbox_inches="tight")
    plt.close('all')

def contour_plot(field, t_step):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.3, wspace=0.25)
    size_f = 20
    pad_f = 20

    for k in range(2):

        ax = fig.add_subplot(2, 1, k + 1)

        im = ax.imshow(field[k][t_step, :, :].T, origin='upper', cmap='jet', vmin =-.4, vmax = .4)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.8)

        if k == 1:
            ax.set_title(r'$ROM$, t={}'.format(t_step), rotation=0, size=size_f, pad=pad_f)

        else:
            ax.set_title(r'$FOM$, t={}'.format(t_step), rotation=0, size=size_f, pad=pad_f)

        colorbar = plt.colorbar(im, cax=cax, ticks=np.linspace(-0.4, 0.4, 6)) ## I have no idea why this doesn't always have the same # of ticks
        colorbar.ax.set_title(r'$m/s$', rotation=0, size=size_f, pad=pad_f)
        colorbar.ax.tick_params(labelsize=size_f)

        plt.setp(ax.spines.values(), linewidth=2)

    filename = './out/DEFAULT/Reconstruction_t{}.png'.format(t_step + 1)
    plt.savefig(filename)

def probe_plot(uX_probeLES, uX_probeNODE):
    fig, ax = plt.subplots(figsize=(40, 15))
    ax.plot(np.arange(0, 300), uX_probeLES, 'r', linewidth=6, label='LES')
    ax.plot(np.arange(0, 300), uX_probeNODE, 'k:', linewidth=6, label='NODE')
    ax.axvline(x=100, color='k', linewidth=5)

    plt.setp(ax.spines.values(), linewidth=4)
    ax.tick_params(axis='both', which='major', labelsize=70)
    ax.tick_params(axis='both', which='minor', labelsize=70)
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)
    ax.set_ylim(-0.25, 0.25)
    ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, borderaxespad=0., fontsize=80)
    ax.set_ylabel(r'$u^{\prime}_x$', rotation=0, size=80, labelpad=30)
    ax.set_xlabel(r'$t$', size=60)
    filename = './results/probe.pdf'
    plt.savefig(filename, format="pdf", bbox_inches="tight")

def plotLSTM(predLSTM, labelPOD, res_folder):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    fig.suptitle('Reconstruction of POD temporal modes using LSTM', fontsize=24)

    filename = res_folder + '/results_valid'
    t_steps = np.linspace(75, 100, 25)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.35, wspace=0.25)
    
    # iterate over the 8 modes
    for k in range(8):

        ax = fig.add_subplot(5, 2, k + 1)
        ax.plot(t_steps, labelPOD[:, k], color='r', linewidth=2.5, label='POD')
        ax.plot(t_steps, predLSTM[:, k], 'k--', linewidth=2.5, label='LSTM')

        ax.set_ylabel('$a_{%d}$' % (k + 1), rotation=0, size=25, labelpad=10)

        if k == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=25)
        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

    plt.savefig("%s.pdf" % filename, format="pdf", bbox_inches="tight")
    plt.close('all')
