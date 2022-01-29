from matplotlib import tight_layout
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from sci.lib.loader import *
from sci.lib.vis.reconstruct import *


# SETTINGS
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
DPI = 160

REC = {'VKS':vks_reconstruct, 'KPP':kpp_reconstruct, 'EE':ee_reconstruct}

def dmd_modes(dataloader,args):
    DL = dataloader

    if args.dataset == 'VKS':
        fig = plt.figure(figsize=(3,15), tight_layout=True, dpi=DPI)
        for i in range(4):
            ax = plt.subplot(args.modes,2,i*2+1)
            # ax.imshow(np.absolute(DL.data[0,:,:,i].T))
            REC[args.dataset](np.absolute(DL.data),i,ax,args,index=0)
            ax.set_axis_off()
            ax = plt.subplot(args.modes,2,i*2+2)
            REC[args.dataset](np.absolute(DL.data),i,ax,args,index=1)
            ax.set_axis_off()

    elif args.dataset == 'KPP':
        fig = plt.figure(figsize=(3,15), tight_layout=True, dpi=DPI)
        for i in range(4):
            ax = plt.subplot(args.modes,2,i*2+1, projection='3d')
            REC[args.dataset](np.absolute(DL.data),i,ax,args,index=0)
            ax.set_axis_off()
            ax = plt.subplot(args.modes,2,i*2+2, projection='3d')
            REC[args.dataset](np.absolute(DL.data),i,ax,args,index=1)
            ax.set_axis_off()

    elif args.dataset == 'EE':
        fig = plt.figure(figsize=(3,15), tight_layout=True, dpi=DPI)
        for i in range(4):
            ax = plt.subplot(args.modes,3,i*3+1)
            REC[args.dataset](np.absolute(DL.data),i,ax,0,args)
            ax.set_axis_off()
            ax = plt.subplot(args.modes,3,i*3+2)
            REC[args.dataset](np.absolute(DL.data),i,ax,1,args)
            ax.set_axis_off()
            ax = plt.subplot(args.modes,3,i*3+3)
            REC[args.dataset](np.absolute(DL.data),i,ax,2,args)
            ax.set_axis_off()

    return fig