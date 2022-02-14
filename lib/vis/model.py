#IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append('./')

from lib.datasets import * 
from lib.utils import *
from lib.vis import *

############################
### MODEL UNIVERSAL PLOTS ###
############################

"""
INNER METHODS
"""
def ax_nfe(epochs,nfes,plt_args):
  plt.scatter(epochs,nfes,**plt_args)
  return 1

def ax_loss(loss,plt_args):
  plt.plot(loss,**plt_args)
  return 1




"""
SINGLES
"""


"""  LOSS PLOT """
def plot_loss(fname,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['tr_loss', 'val_loss']
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
def plot_nfe(fname,index_,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)

    nfes = df[index_].values
    epochs=np.arange(len(nfes))
    plt.figure(tight_layout=True)
    plt.scatter(epochs,nfes)
    end_str = str(args.out_dir+'/'+args.model+'_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

"""  ADJOINT GRADIENT PLOT """
def plot_adjGrad(fname,args,show=False):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['grad_{}'.format(i) for i in range(args.seq_ind)]
    grad = df[index_].values
    plt.figure(tight_layout = True)
    plt.imshow(grad.T, vmin=0, vmax = .05, cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.title(args.model+' Adjoint Gradient')
    plt.xlabel('Epoch')
    plt.ylabel('$T-t$')
    
    end_str = str(args.out_dir+'/'+args.model+'_adjGrad')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1

"""  STIFFNESS PLOT """
def plot_stiff(fname,args, clip=1, show=False):
    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['backward_stiff']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yscale('log')

    end_str = str(args.out_dir+'/'+args.model+'_stiffness')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return 1


""""
COMPARISONS
"""
def compare_nfe(file_list,model_list,index_,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        nfes = df[index_].values[::args.epoch_freq]
        epochs=np.arange(len(nfes))
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        ax_nfe(epochs,nfes,plt_args)

    plt.legend()
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()


def compare_loss(file_list,model_list,index_,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        losses = df[index_].values
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        ax_loss(losses[::args.epoch_freq],plt_args)

    plt.yscale('log')
    plt.legend()
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
