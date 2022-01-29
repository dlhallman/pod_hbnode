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
from lib.datasets import * 
from lib.utils import *
from lib.vis.reconstruct import *
from lib.models.seq import *

# SETTINGS
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
DPI = 160

############################
##### DATA BASED PLOTS #####
############################

"""  EIGEN VALUE DECAY PLOT """
def eig_decay(dataloader,args):
    #INITIALIZE
    DL = dataloader
    total = DL.lv.sum()
    decay=[1]
    true_switch = False #prevent overflow that occurs from dividing by total
    #CALCULATE DECAY
    # for eig in DL.lv[:args.modes]:
    for eig in DL.lv:
        if eig < 1e-14:
            true_switch=True
        if true_switch:
            val = 0
        else:
            val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    x = np.arange(0,len(DL.lv)+1)
    #GENERATE-Y TICKS
    yticks = []
    i=0
    power = 1/min(decay[np.where(decay>0)])
    while(10**(i-1)<power):
        yticks=yticks+[10**(-i)]
        i=i+1
    num = len(yticks)//5

    plt.figure(tight_layout=True, dpi=DPI)
    plt.plot(x,decay, 'k')
    #AXES
    plt.yscale('log')
    plt.yticks(yticks[::num])
    # num = (args.modes-1)//5
    num = (len(DL.lv)+1)//5
    plt.xticks(list(x[::num])+[x[-1]])
    #TITLES
    plt.xlabel('Mode ($r$)')
    plt.ylabel('Decay ($\\varepsilon$)')
    #OUTPUT
    plt.savefig(args.out_dir+'/'+args.model+'/eig_decay.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    print("Decay value is {} for {} modes.".format(args.modes,decay[args.modes]))
    
    return 1

""" DATA MODE PLOTS """
def data_modes(dataloader,args):

    DL = dataloader

    if args.model == "DMD":
        fig = dmd_modes(DL,args)
    else:
        fig = plt.figure(figsize=(10,15), tight_layout=True, dpi=DPI)
        for i in range(4):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(DL.data[:,i], 'k')
            plt.xlabel("$t$")
            plt.ylabel("$\\alpha_{}$".format(i))

    plt.savefig(args.out_dir+'/'+args.model+'/full_modes.pdf', format="pdf", bbox_inches="tight")

    if args.verbose: plt.show()

    return 1


############################
### DATA RECONSTRUCTIONS ###
############################

REC = {'VKS':vks_reconstruct, 'KPP':kpp_reconstruct, 'EE':ee_reconstruct}
def data_reconstruct(data,time,args,heat=False):

    fig = plt.figure(figsize=(10,10), tight_layout=True)
    if args.dataset == 'VKS':
        ax = plt.subplot(211)
        REC[args.dataset](data,time,ax,args,index=0,heat=heat)
        ax.set_title('$u\'_x$')
        ax = plt.subplot(212)
        REC[args.dataset](data,time,ax,args,index=1,heat=heat)
        ax.set_title('$u\'_y$')
    elif args.dataset == 'KPP':
        ax = fig.add_subplot(projection='3d')
        REC[args.dataset](data,time,ax,args)
    elif args.dataset=='EE':
        ax = plt.subplot(311)
        REC[args.dataset](data,time,ax,0,args)
        ax = plt.subplot(312)
        REC[args.dataset](data,time,ax,1,args)
        ax = plt.subplot(313)
        REC[args.dataset](data,time,ax,2,args)


    if args.verbose: plt.show()
    plt.savefig(args.out_dir+'/'+args.model+'/data_recon.pdf', format="pdf", bbox_inches="tight")

    return fig


ANIM = {'VKS':vks_animation, 'KPP':kpp_animation, 'EE':ee_animation}
def data_animation(data,args):

    ani = ANIM[args.dataset](data,args)
    ani.save(args.out_dir+'/'+args.model+'/ANIM.gif', "PillowWriter", fps=5)
    if args.verbose: plt.show()

    return 1