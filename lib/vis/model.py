#IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('../')
from lib.datasets import * 
from lib.utils import *
from lib.vis import *

############################
### MODEL UNIVERSAL PLOTS ###
############################
"""  LOSS PLOT """
def plot_loss(fname,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['loss', 'va_loss']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yscale('log')
    end_str = str(args.out_dir+'/'+args.model+'_loss')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

"""  NFE PLOT """
def plot_nfe(fname,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['forward_nfe']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    end_str = str(args.out_dir+'/'+args.model+'_nfe')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

"""  ADJOINT GRADIENT PLOT """
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

"""  STIFFNESS PLOT """
def plot_stiff(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['backward_stiff']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    
    plt.savefig(args.out_dir+'/'+args.model+'/Stiffness.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1
