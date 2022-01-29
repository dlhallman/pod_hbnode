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


def param_Loss(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['loss', 'ts_loss']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True, dpi=DPI)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yscale('log')
    # yticks = [100/(10**i) for i in range(5)]
    # plt.yticks(yticks)
    plt.savefig(args.out_dir+'/'+args.model+'/LOSS.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1


def param_ModesLong(dataloader, model, args, show=False):
    DL = dataloader
    times = DL.train_times
    stamps = [args.tstart, args.tr_win+args.tstart, args.tstop]
    xs = [np.arange(0,args.tr_win), np.arange(stamps[1], stamps[2]-1)]
    data = DL.train_data[:,:,:]
    labels = DL.train_label[:,:,:]
    plt.figure(figsize=(15,5), tight_layout=True, dpi=DPI)

    predict = model(times, data).cpu().detach().numpy()[:,0,:]

    for i,node in enumerate(predict.T):
        plt.subplot(args.modes//2,2,i+1)
        plt.plot(xs[0],node, 'r', label='Prediction')
        plt.plot(xs[0],labels[:,0,i], 'k--', label='True')
        # plt.axvspan(stamps[1]-2, stamps[1]+args.seq_win, facecolor='k', alpha=.25)
        # plt.axvspan(stamps[0], stamps[0]+args.seq_win, facecolor='k', alpha=.25)
        plt.xlabel("Time")
        plt.ylabel("$\\alpha_{}$".format(i))
        # plt.xlim(stamps[0],stamps[2]-1)
        # plt.title('Mode {}'.format(i+1))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
    plt.savefig(args.out_dir+'/'+args.model+'/'+'modeReconstruct.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    return 1