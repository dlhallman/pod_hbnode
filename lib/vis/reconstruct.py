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
from sci.lib.loader import * 
from sci.lib.utils import *
from sci.lib.vis import *
from sci.lib.seq.models import *
import sci.lib.seq.parser as wparse



def plot_Reconstruction(dataloader, model, args, show=False):

    if args.dataset != 'VKS':
        return 0

    DL = dataloader
    t_ind = args.tr_win - 2
    Nxy = DL.data_init.shape[0] * DL.data_init.shape[1]
    times = [DL.train_times, DL.valid_times, DL.eval_times]
    index = [np.arange(1+args.tstart,args.tstart+args.tr_win), np.arange(1+args.tstart+args.tr_win, args.tstart+args.val_win), np.arange(1+args.tstart+args.tr_win,DL.eval_times.shape[1])]
    data = [DL.train_data, DL.valid_data, DL.eval_data]
    data_x = DL.ux[index[0],:,:]
    data_y = DL.uy[index[0],:,:]
    # pod_true = DL.seq_data[0][1:,:]
    pod_true = DL.train_label
    predict = model(times[0], data[0]).cpu().detach().numpy()[-1,:-1,:]
    spatial_modes = DL.spatial_modes

    pod_true = (pod_true + DL.mean)*DL.std
    predict = (predict + DL.mean)*DL.std

    pod_true = np.matmul(pod_true, spatial_modes.T)
    predict = np.matmul(predict, spatial_modes.T)

    pod_x = pod_true[-1,:, :Nxy]
    pod_y = pod_true[-1,:, Nxy:]
    predict_x = predict[:, :Nxy]
    predict_y = predict[:, Nxy:]

    shape = [pod_true.shape[0], DL.data_init.shape[0], DL.data_init.shape[1]]

    pod_x = pod_x.reshape(pod_x.shape[0], shape[1], shape[2])
    pod_y = pod_y.reshape(pod_y.shape[0], shape[1], shape[2])
    predict_x = predict_x.reshape(predict_x.shape[0], shape[1], shape[2])
    predict_y = predict_y.reshape(predict_y.shape[0], shape[1], shape[2])

    plt.figure(figsize=(10,10), tight_layout=True)
    plt.subplot(231)
    plt.imshow(data_x[t_ind,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title('FOM U_x')
    plt.subplot(232)
    plt.imshow(pod_x[-1,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title('ROM U_x')
    plt.subplot(233)
    plt.imshow(predict_x[-1,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title(args.model+' U_x')
    plt.subplot(234)
    plt.imshow(data_y[t_ind,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title('FOM U_y')
    plt.subplot(235)
    plt.imshow(pod_y[-1,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title('ROM U_y')
    plt.subplot(236)
    plt.imshow(predict_y[-1,:,:], origin='upper', cmap='jet', vmin =-.4, vmax = .4)
    plt.title(args.model+' U_y')
    plt.savefig(args.out_dir+'/'+args.model+'/RECONSTRUCTION.pdf', format="pdf", bbox_inches="tight")

    
    if args.verbose: plt.show()
    return 1



def plot_Anim(dataloader, model, args, show=False):

    if args.dataset != 'VKS':
        return 0

    DL = dataloader
    Nxy = DL.data_init.shape[0] * DL.data_init.shape[1]
    times = [DL.train_times, DL.valid_times, DL.eval_times]

    end = min(DL.data_init.shape[2],args.tstop)
    index = [np.arange(1+args.tstart,args.tstart+args.tr_win), np.arange(1+args.tstart+args.tr_win, args.tstart+args.val_win), np.arange(1+args.tstart+args.val_win,end-args.tstart)]
    data = [DL.train_data, DL.valid_data, DL.eval_data]
    data_x = np.vstack((DL.ux[index[0],:,:],DL.ux[index[1],:,:]))
    data_x = np.vstack((data_x,DL.ux[index[2],:,:]))
    data_y =np.vstack(( DL.uy[index[0],:,:],DL.uy[index[1],:,:]))
    data_y =np.vstack((data_y,DL.uy[index[2],:,:]))
    # pod_true = DL.seq_data[0][1:,:]
    pod_true = DL.seq_data[0][DL.seq_win:,:]
    predict = model(times[0], data[0]).detach().cpu().numpy()[-1,:-1,:]
    predict = np.vstack((predict, model(times[1], data[1]).detach().cpu().numpy()[-1,:-1,:]))
    spatial_modes = DL.spatial_modes

    pod_true = np.matmul(pod_true, spatial_modes.T)
    predict = (predict+DL.mean)*DL.std
    predict = np.matmul(predict, spatial_modes.T)

    pod_x = pod_true[:, :Nxy]
    pod_y = pod_true[:, Nxy:]
    predict_x = predict[:, :Nxy]
    predict_y = predict[:, Nxy:]

    shape = [pod_true.shape[0], DL.data_init.shape[0], DL.data_init.shape[1]]

    pod_x = pod_x.reshape(pod_x.shape[0], shape[1], shape[2])
    pod_y = pod_y.reshape(pod_x.shape[0], shape[1], shape[2])
    predict_x = predict_x.reshape(predict_x.shape[0], shape[1], shape[2])
    predict_y = predict_y.reshape(predict_y.shape[0], shape[1], shape[2])

    fig, axes = plt.subplots(2,3, figsize=(15,8), tight_layout=True)
    lines = []
    for ax in axes.flatten():
        lines = lines + [ax.imshow(np.zeros((shape[0],shape[1])), origin='upper', cmap='jet', vmin =-.4, vmax = .4)]

    def run(vks_t):

        lines[0].set_data(data_x[vks_t,:,:].T)
        lines[1].set_data(pod_x[vks_t,:,:].T)
        lines[2].set_data(predict_x[vks_t,:,:].T)
        lines[3].set_data(data_y[vks_t,:,:].T)
        lines[4].set_data(pod_y[vks_t,:,:].T)
        lines[5].set_data(predict_y[vks_t,:,:].T)

        return lines


    ani = animation.FuncAnimation(fig, run, blit=True, interval=end-1,
        repeat=False)

    ani.save(args.out_dir+'/'+args.model+'/ANIM.gif', "PillowWriter", fps=5)
    if args.verbose: plt.show()
    return 1
