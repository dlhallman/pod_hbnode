#IMPORTS
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys

plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 28
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
DPI = 160


#######################
### RECONSTRUCTIONS ###
#######################
def vks_plot(data,time,axis,args,index=None):
    plt.style.use('classic')
    axis.imshow(data[time,:,:,index].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
    return 1

def kpp_plot(data,time,axis,args,index=None):
    plt.style.use('default')
    xv =  np.tile(np.linspace(-2,2,data.shape[1]),(data.shape[2],1))
    yv = np.tile(np.linspace(-2.4,1.4,data.shape[2]),(data.shape[1],1)).T
    axis.plot_surface(xv, yv, data[time], cmap=cm.coolwarm, linewidth=0)
    return 1

def ee_plot(data,time,axis,index,args, heat=None):
    x = np.linspace(-5,5,data.shape[1])
    axis.plot(x,data[index,:,time], 'k')
    return 1

"""
DATA RECONSTRUCTION HEAD
"""

def data_reconstruct(data,time,args):

    fig = plt.figure(tight_layout=True)
    if args.dataset == 'VKS':
        ax = plt.subplot(211)
        vks_plot(data,time,ax,args,index=0)
        ax.set_title('$u\'_x$')
        ax = plt.subplot(212)
        vks_plot(data,time,ax,args,index=1)
        ax.set_title('$u\'_y$')
    elif args.dataset == 'KPP':
        ax = fig.add_subplot(projection='3d')
        kpp_plot(data,time,ax,args)
    elif args.dataset=='EE':
        ax = plt.subplot(311)
        ee_plot(data,time,ax,0,args)
        ax = plt.subplot(312)
        ee_plot(data,time,ax,1,args)
        ax = plt.subplot(313)
        ee_plot(data,time,ax,2,args)

    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    return fig
