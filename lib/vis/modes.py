#IMPORTS
from matplotlib import tight_layout
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import sys
import torch

sys.path.append('../../')
from lib.datasets import * 
from lib.utils import *
from lib.vis import *
from lib.seq.models import *
import sci.lib.seq.parser as wparse

#PLOT FORMATTING
plt.rcParams['font.family'] = 'Times New Roman'
DPI = 160

def plot_Modes(dataloader, model, args, show=False):
    DL = dataloader
    names = ['TRAIN', 'VALID', 'EVAL']
    fullNames = ['Traning', 'Validation', 'Evaluation']
    times = [DL.train_times, DL.valid_times, DL.eval_times]
    stamps = [args.tstart, args.tr_win+args.tstart, args.val_win+args.tstart, args.tstop]
    # xs = [np.arange(stamps[0]+args.seq_win,stamps[1]), np.arange(stamps[1]+args.seq_win, stamps[2]), np.arange(stamps[2]+args.seq_win,stamps[3])]
    xs = [np.arange(stamps[0]+args.seq_win,stamps[1]-1), np.arange(stamps[1]+args.seq_win, stamps[2]-1), np.arange(stamps[2]+args.seq_win,stamps[3]-1)]
    data = [DL.train_data, DL.valid_data, DL.eval_data]
    labels = [DL.train_label, DL.valid_label, DL.eval_label]
    for j in range(3):
        predict = model(times[j], data[j]).cpu().detach().numpy()[-1,:-1,:args.print_modes]
        actual = data[j][-1,1:,:args.print_modes]

        plt.figure(figsize=(15,5), tight_layout=True, dpi=DPI)
        plt.suptitle("POD {} Modes".format(fullNames[j]))
        for i,node in enumerate(predict.T):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(xs[j],node, 'r', label='Prediction')
            plt.plot(xs[j],actual.cpu().T[i], 'k--', label='True')
            plt.xlabel("Time")
            plt.ylabel("$\\alpha_{}$".format(i))
            # plt.title('Mode {}'.format(i+1))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
        plt.savefig(args.out_dir+'/'+args.model+'/'+names[j]+'.pdf', format="pdf", bbox_inches="tight")
        if args.verbose: plt.show()

    return 1

def plot_ModesLong(dataloader, model, args, show=False):
    DL = dataloader
    times = [DL.train_times, DL.valid_times]
    stamps = [args.tstart, args.tr_win+args.tstart, args.val_win+args.tstart]
    xs = [np.arange(stamps[0]+args.seq_win,stamps[1]-1), np.arange(stamps[1]+args.seq_win, stamps[2]-1)]
    data = [DL.train_data, DL.valid_data]
    labels = [DL.train_label, DL.valid_label]
    plt.figure(figsize=(15,5), tight_layout=True, dpi=DPI)
    for j in range(2):
        predict = model(times[j], data[j]).cpu().detach().numpy()[-1,:-1,:args.print_modes]
        actual = data[j][-1,1:,:args.print_modes]

        for i,node in enumerate(predict.T):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(xs[j],node, 'r', label='Prediction')
            plt.plot(xs[j],actual.cpu().T[i], 'k--', label='True')
            plt.axvspan(stamps[1]-2, stamps[1]+args.seq_win, facecolor='k', alpha=.25)
            plt.axvspan(stamps[0], stamps[0]+args.seq_win, facecolor='k', alpha=.25)
            plt.xlabel("Time")
            plt.ylabel("$\\alpha_{}$".format(i))
            plt.xlim(stamps[0],stamps[2]-1)
            # plt.title('Mode {}'.format(i+1))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
    plt.savefig(args.out_dir+'/'+args.model+'/'+'modeReconstruct.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    return 1