#IMPORTS
from einops import rearrange
import numpy as np
import os
import pandas as pd
import pickle
import torch
from sci.lib.pod import *

#STANDARD DATA RETREIVAL
def VKS_DAT(data_dir, param=None):
    """Loads Von Karman Flow Data Set"""
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    data = np.nan_to_num(data)
    return data

def EE_DAT(data_dir, param):
    """Loads Euler Equation Data Set"""
    npzdata = np.load(data_dir)
    rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
    return np.array([rho[:,:,param], u[:,:,param], E[:,:,param]], dtype=np.double)

def FIB_DAT(data_dir, param):
    data = pd.read_table(data_dir, sep="\t", index_col=2, names=["x", "h"]).to_numpy()
    end = data.shape[0]//401
    return data[:,1].reshape(end,401)


def KPP_DAT(data_dir, param):
    npdata = np.load(data_dir)
    xv, yv, kpp = npdata['arr_0'], npdata['arr_1'], npdata['arr_2']
    return kpp

#PARAMETERIZED DATA
def EE_PARAM(data_dir):
    """Loads Euler Equation Data Set"""
    npzdata = np.load(data_dir)
    rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
    return np.array([rho, u, E], dtype=np.double)

LOADERS = {'VKS':VKS_DAT, 'EE': EE_DAT, 'FIB' : FIB_DAT, 'KPP': KPP_DAT}



#EMPTY DATASET CLASS
class DataSet:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


class DMD_LOADER():
    def __init__(self, args, modes=None):
        
            #ARGS
            assert args.dataset in LOADERS
            #DATA CONFIGS
            self.dataset = args.dataset
            self.data_dir = args.data_dir
            self.modes = args.modes
            self.tstart = args.tstart
            self.tstop = args.tstop
            #SPLIT CONFIGS
            self.tr_ind = args.tr_ind
            self.val_ind = args.val_ind

            print('Loading ... \t Dataset: {}'.format(self.dataset))
            self.data_init = LOADERS[self.dataset](args.data_dir, args.paramEE)
            self.data = self.data_init

            if self.modes != None:
                self.reduce()


    def reduce(self):

        # DMD REDUCTION
        return 0


class STD_LOADER():
    def __init__(self, args, modes=None):
    
        #ARGS
        assert args.dataset in LOADERS
        #DATA CONFIGS
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.modes = args.modes
        self.tstart = args.tstart
        self.tstop = args.tstop
        #SPLIT CONFIGS
        self.tr_ind = args.tr_ind
        self.val_ind = args.val_ind

        print('Loading ... \t Dataset: {}'.format(self.dataset))
        self.data_init = LOADERS[self.dataset](args.data_dir, args.paramEE)
        self.data = self.data_init

        if self.modes != None:
            self.reduce()
        
        #DATA SPLITTNG
        train_data = self.data[:self.tr_ind, :]
        valid_data = self.data[:self.val_ind, :]

        #DATA NORMALIZATION
        self.mean_data = train_data.mean(axis=0)
        self.std = train_data.std(axis=0)

        train_data = (train_data - self.mean_data) / self.std
        valid_data = (valid_data - self.mean_data) / self.std

        #TENSOR DATA
        train_data = train_data.reshape((1, train_data.shape[0], train_data.shape[1]))
        self.train_data = torch.FloatTensor(train_data)

        valid_data = valid_data.reshape((1, valid_data.shape[0], valid_data.shape[1]))
        self.valid_data = torch.FloatTensor(valid_data)

        self.data_eval = self.data[self.val_ind:, :]

        #INVERT OVER TIME
        idx = [i for i in range(self.train_data.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        self.obs_t = self.train_data.index_select(0, idx)

        #NORMALIZE TIME
        eval_times = np.linspace(0, 1, self.data.shape[0])
        self.eval_times = torch.from_numpy(eval_times).float()
        self.train_times = self.eval_times[:self.tr_ind]
        self.valid_times = self.eval_times[:self.val_ind]

    def reduce(self):
        """POD Model Reduction"""
        print('Reducing ... \t Modes: {}'.format(self.modes))

        #POD CALL
        if self.dataset == 'FIB':
            self.spatial_modes, self.data, self.lv, self.ux = POD1(self.data, self.tstart, self.tstop, self.modes)
        if self.dataset == 'VKS':
            self.spatial_modes, self.data, self.lv, self.ux, self.uy = POD2(self.data, self.tstart, self.tstop, self.modes)
        elif self.dataset == 'EE':
            self.spatial_modes, self.data, self.lv, self.ux, self.uy, self.uz = POD3(self.data, self.tstart, self.tstop, self.modes)
        elif self.dataset == 'KPP': #flatten and use POD1
            self.spatial_modes, self.data, self.lv, self.ux = PODKPP(self.data, self.tstart, self.tstop, self.modes)
        
        # recovery_per = eigenvalues[:pod_modes] / eigenvalues.sum() * 100

        return 1

class SEQ_LOADER:

    def __init__(self, args):
        
        assert args.dataset in LOADERS
        #DATA CONFIG
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.modes = args.modes
        self.tstart = args.tstart
        self.tstop = args.tstop
        #SPLIT CONFIG
        self.batch_size = args.batch_size
        self.seq_win = args.seq_win
        self.tr_win = args.tr_win
        self.val_win = args.val_win
        self.device = args.device

        print('Loading ... \t Dataset: {}'.format(self.dataset))
        self.data_init = LOADERS[self.dataset](args.data_dir, args.paramEE)
        self.data = self.data_init

        if self.modes != None:
            self.reduce()
        total_size = self.data.shape[0] - self.seq_win
        args.tstop = min(args.tstop, self.data.shape[0]+args.tstart-1)
        
        #SEQUENCE DATA
        total_size = self.data.shape[0] - self.seq_win
        #SEQUENCE DATA
        seq_data = np.vstack([[self.data[t:t + self.seq_win, :] for t in range(total_size-1)]]).swapaxes(0,1)
        seq_label = np.vstack([[self.data[t+1:t+self.seq_win+1, :] for t in range(total_size-1)]]).swapaxes(0,1)
        tr_win = self.tr_win
        val_win = self.val_win
        self.seq_data = seq_data
        self.seq_label = seq_label
                
        # training data
        train_data = seq_data[:, :tr_win-self.seq_win, :]
        train_label = seq_label[:, :tr_win-self.seq_win, :]
        self.train_data =  torch.FloatTensor(train_data)
        self.mean = train_data.reshape((-1, train_data.shape[2])).mean(axis=0)
        self.std = train_data.reshape((-1, train_data.shape[2])).std(axis=0)
        self.train_data = torch.FloatTensor((train_data - self.mean) / self.std).to(self.device)


        self.train_label = torch.FloatTensor(train_label)
        self.train_label = torch.FloatTensor((train_label - self.mean) / self.std).to(self.device)
        self.train_times = (torch.ones(train_data.shape[:-1])/train_data.shape[1]).to(self.device)

        # validation data
        val_data = (seq_data[:, tr_win:val_win-self.seq_win, :]-self.mean)/self.std
        val_label = (seq_label[:, tr_win:val_win-self.seq_win, :]-self.mean)/self.std
        self.valid_data =  torch.FloatTensor(val_data).to(self.device)
        self.valid_label = torch.FloatTensor(val_label).to(self.device)
        self.valid_times = (torch.ones(val_data.shape[:-1])/val_data.shape[1]).to(self.device)

        # validation data
        eval_data = (seq_data[:, val_win:, :]-self.mean)/self.std
        eval_label = (seq_label[:, val_win:, :]-self.mean)/self.std
        self.eval_data =  torch.FloatTensor(eval_data).to(self.device)
        self.eval_label = torch.FloatTensor(eval_label).to(self.device)
        self.eval_times = (torch.ones(eval_data.shape[:-1])/eval_data.shape[1]).to(self.device)

    def reduce(self):
        """POD Model Reduction"""
        print('Reducing ... \t Modes: {}'.format(self.modes))

        #POD CALL
        if self.dataset == 'FIB':
            self.spatial_modes, self.data, self.lv, self.ux = POD1(self.data, self.tstart, self.tstop, self.modes)
        if self.dataset == 'VKS':
            self.spatial_modes, self.data, self.lv, self.ux, self.uy = POD2(self.data, self.tstart, self.tstop, self.modes)
        elif self.dataset == 'EE':
            self.spatial_modes, self.data, self.lv, self.ux, self.uy, self.uz = POD3(self.data, self.tstart, self.tstop, self.modes)
        elif self.dataset == 'KPP': #flatten and use POD1
            self.spatial_modes, self.data, self.lv, self.ux = PODKPP(self.data, self.tstart, self.tstop, self.modes)


class PARAM_LOADER:

    def __init__(self, args):
        
        assert args.dataset == "EE"
        #DATA CONFIG
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.modes = args.modes
        self.tstart = args.tstart
        self.tstop = args.tstop
        #SPLIT CONFIG
        self.batch_size = args.batch_size
        self.tr_win = args.tr_win
        self.device = args.device

        print('Loading ... \t Dataset: {}'.format(self.dataset))
        self.data_init = EE_PARAM(args.data_dir)
        self.data = self.data_init
        self.params = self.data_init.shape[-1]

        if self.modes != None:
            self.reduce()
        args.tstop = min(args.tstop, self.data.shape[0]+args.tstart-1)
        
        #SEQUENCE DATA
        rev = self.tstop - (self.tstart + self.tr_win)
        train_size = 1 # int(0.8 * self.params)
        self.train =self.data[:train_size]
        self.eval =self.data[train_size:2] #remvoe 2 and pervious trainsize increase


        #SEQUENCE DATA
        train_data = self.train[:,:self.tr_win,:].swapaxes(0,1)
        train_label = self.train[:,rev:,:].swapaxes(0,1)
        self.mean = train_data.reshape((-1, train_data.shape[2])).mean(axis=0)
        self.std = train_data.reshape((-1, train_data.shape[2])).std(axis=0)
        self.train_data = torch.FloatTensor((train_data - self.mean) / self.std)

        self.train_label = torch.FloatTensor(train_label).to(self.device)
        self.train_label = torch.FloatTensor((train_label - self.mean) / self.std)
        self.train_times = (torch.ones(train_data.shape[:-1])/train_data.shape[1]).to(self.device)

        eval_data = self.eval[:,:self.tr_win, :].swapaxes(0,1)
        eval_label = self.eval[:,rev:, :].swapaxes(0,1)
        self.eval_data =  torch.FloatTensor(eval_data).to(self.device)
        self.eval_label = torch.FloatTensor(eval_label).to(self.device)
        self.eval_times = (torch.ones(eval_data.shape[:-1])/eval_data.shape[1]).to(self.device)


    def reduce(self):
        """POD Model Reduction"""
        print('Reducing ... \t Modes: {}'.format(self.modes))

        #POD CALL
        if self.dataset == 'EE':
            self.spatial_modes = []
            self.temp = []
            self.lv = []
            self.ux = []
            self.uy = []
            self.uz = []
            for i in range(self.params):
                spatial_modes, temp, lv, ux, uy, uz = POD3(self.data[:,:,:,i], self.tstart, self.tstop, self.modes)
                self.spatial_modes = self.spatial_modes + [spatial_modes]
                self.temp = self.temp + [temp]
                self.lv = self.lv + [lv]
                self.ux = self.ux + [ux]
                self.uy = self.uy + [uy]
                self.uz = self.uz + [uz]

            self.spatial_modes = np.array(self.spatial_modes)
            self.data = np.array(self.temp)
            self.lv = np.array(self.lv)
            self.ux = np.array(self.ux)
            self.uy = np.array(self.uy)
            self.uz = np.array(self.uz)