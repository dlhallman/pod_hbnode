#IMPORTS
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import sys
import torch

sys.path.append('../')
from sci.lib.loader import * 
from sci.lib.utils import *
from sci.lib.vis import *
from sci.lib.seq.models import *
import sci.lib.seq.parser as wparse

# SETTINGS
plt.rcParams['font.family'] = 'Times New Roman'
DPI = 160

def plot_AdjGrad(fname,args, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['grad_{}'.format(i) for i in range(args.seq_win)]
    grad = df[index_].values
    plt.figure(tight_layout = True, dpi=DPI)
    plt.imshow(grad.T, vmin=0, vmax = .05, cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.title(args.model+' Adjoint Gradient')
    plt.xlabel('Epoch')
    plt.ylabel('$T-t$')
    plt.savefig(args.out_dir+'/'+args.model+'/AdjGrad.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

def plot_Loss(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['loss', 'va_loss']
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

def plot_NFE(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['va_nfe']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True, dpi=DPI)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.savefig(args.out_dir+'/'+args.model+'/NFE.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

def plot_stiff(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['backward_stiff']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True, dpi=DPI)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.savefig(args.out_dir+'/'+args.model+'/Stiffness.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1