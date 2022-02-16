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

def ax_stiff(epochs,stiff,plt_args):
  plt.scatter(epochs,stiff,**plt_args)
  return 1

def ax_loss(epochs,loss,plt_args):
  plt.plot(epochs,loss,**plt_args)
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
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['tr_loss', 'val_loss']
    color = ['k','r--']
    losses = df[index_].values
    epochs=np.arange(len(losses))
    plt.figure(tight_layout=True)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yticks(np.logspace(-4,0,5))
    plt.ylim(1e-4,1)
    plt.xlim(epochs[0],epochs[-1])
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
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)

    nfes = df[index_].values
    epochs=np.arange(len(nfes))
    plt.figure(tight_layout=True)
    plt.scatter(epochs,nfes)
    plt.xlim(epochs[0],epochs[-1])
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
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['grad_{}'.format(i) for i in range(args.seq_ind)]
    grad = df[index_].values
    plt.figure(tight_layout = True)
    plt.imshow(grad.T, origin='upper',vmin=0,vmax=.01, cmap='inferno', aspect='auto')
    plt.colorbar()
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
    stiff = df[index_].values
    epochs = np.arange(len(stiff))
    plt.figure(tight_layout=True)
    plt.scatter(epochs,stiff)
    plt.yscale('log')
    plt.xlim(0,epochs[-1])

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
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        nfes = df[index_].values[::args.epoch_freq]
        epochs=np.arange(len(nfes))*args.epoch_freq
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        ax_nfe(epochs,nfes,plt_args)

    plt.xlim(epochs[0],epochs[-1])
    plt.legend()
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

def compare_stiff(file_list,model_list,index_,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        stiffs = df[index_].values[::args.epoch_freq]
        epochs=np.arange(len(stiffs))*args.epoch_freq
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        ax_stiff(epochs,stiffs,plt_args)

    plt.legend() 
    plt.ylim(1,1e4)
    plt.yscale('log')
    plt.xlim(epochs[0],epochs[-1])
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

def compare_loss(file_list,model_list,index_,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        losses = df[index_].values
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        epochs=np.arange(len(losses[::args.epoch_freq]))*args.epoch_freq
        ax_loss(epochs,losses[::args.epoch_freq],plt_args)

    if index_=='tr_loss':
        plt.yticks(np.logspace(-4,0,5))
        plt.ylim(1e-4,1)
    else:
        plt.yticks(np.logspace(-3,0,4))
        plt.ylim(1e-3,1)
    epochs=np.arange(len(losses[::args.epoch_freq]))*args.epoch_freq
    print(epochs)
    plt.xlim(0,epochs[-1])
    plt.yscale('log')
    plt.legend()
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
