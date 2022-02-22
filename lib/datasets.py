#IMPORTS
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from lib.decomp.dmd import *
from lib.decomp.pod import *
from lib.vis.modes import plot_mode

"""STANDARD DATA RETRIEVAL
    - Format dimensions as [t,[omega],k] e.g. [t,[xx,yy,zz],k]
"""
def VKS_DAT(data_file, param=None):
  with open(data_file, 'rb') as f:
    vks = pickle.load(f)
  vks = np.nan_to_num(vks)
  vks = np.moveaxis(vks, 2,0)
  return vks[:,:,:,:2]

def EE_DAT(data_dir, param=0):
  npzdata = np.load(data_dir)
  rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
  ee = np.array([rho[:,:,:param], u[:,:,:param], E[:,:,:param]], dtype=np.double)
  ee = np.moveaxis(ee,1,2)
  ee=ee.reshape(ee.shape[:2]+(-1,1),order='F')[:,:,:,0]
  ee=np.moveaxis(ee,0,-1)
  return ee

def FIB_DAT(data_dir, param=None):
  data = pd.read_table(data_dir, sep="\t", index_col=2, names=["x", "h"]).to_numpy()
  end = data.shape[0]//401
  return data[:,1].reshape(end,401)

def KPP_DAT(data_dir, param=None):
  npdata = np.load(data_dir)
  xv, yv, kpp = npdata['arr_0'], npdata['arr_1'], npdata['arr_2']
  kpp = np.moveaxis(kpp, 2,0)
  return kpp

"""PARAMETERIZED DATA RETRIEVAL
    - Format dimensions as [t,[omega],k,mu] e.g. [t,[xx,yy,zz],k,mu]
"""
def EE_PARAM(data_dir):
  npzdata = np.load(data_dir)
  rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
  ee = np.array([rho, u, E], dtype=np.double)
  return ee.T
LOADERS = {'VKS':VKS_DAT, 'EE': EE_DAT, 'FIB' : FIB_DAT, 'KPP': KPP_DAT}


"""
DATASETS
    - BASE DATASETS
"""
class DMD_DATASET(Dataset):
    def __init__(self, args):
        self.args = args
        assert args.dataset in LOADERS

        print('Loading ... \t Dataset: {}'.format(args.dataset))
        self.data_init = LOADERS[args.dataset](args.data_dir)
        self.data = self.data_init
        self.shape = self.data_init.shape
        self.data_recon=None

        if not args.load_file is None:
            self.load_file(args.load_file)
            self._set_shapes()

    def reduce(self):
        args = self.args
        print('Reducing ... \t Modes: {}'.format(self.args.modes))
        if args.dataset == 'FIB':
            self.X, self.Atilde, self.Ur, self.Phi, self.Lambda, self.lv, self.b = DMD1(self.data, args.tstart, args.tstop, args.modes, lifts=args.lifts)
        elif args.dataset == 'VKS':
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:-1]
            self.component_len = self.shape[-1]
            self.X,self.Atilde,self.Ur,self.Phi,self.Lambda,self.lv,self.b=DMD2(self.data, args.tstart, args.tstop, args.modes, lifts=args.lifts)
            self.data = self.Phi.reshape([self.shape[-1]]+list(self.shape[1:-1])+[len(args.lifts)+1]+[args.modes]).T
        elif args.dataset == 'EE':
            self.component_len = self.shape[-1]
            self.domain_len = self.shape[1]
            self.domain_shape = [self.shape[1]]
            self.X, self.Atilde, self.Ur, self.Phi, self.Lambda, self.lv, self.b = DMD3(self.data, args.tstart, args.tstop, args.modes, lifts=args.lifts)
        elif args.dataset == 'KPP': #flatten and use POD1
            self.component_len = 1
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:]
            self.X, self.Atilde, self.Ur, self.Phi, self.Lambda, self.lv, self.b = DMDKPP(self.data, args.tstart, args.tstop, args.modes, lifts=args.lifts)
        self.time_len = self.X.shape[-1]
        return 1

    def reconstruct(self,time_shape=None):
      args = self.args
      if self.data_recon is None:
          raise Exception('Reconstruction data has not been set')
      if time_shape is None:
          time_shape=args.tpred
      end_shape = [time_shape]+list(self.domain_shape)

      if args.dataset == 'VKS':
          var1_xk = np.real(self.data_recon[:,:self.domain_len].reshape(end_shape))
          var2_xk = np.real(self.data_recon[:,self.domain_len:].reshape(end_shape))
          self.data_recon = np.moveaxis(np.array((var1_xk,var2_xk)),0,-1)
      elif args.dataset == 'EE':
          var1_xk = np.real(self.data_recon[:,:self.domain_len].reshape(end_shape))
          var2_xk = np.real(self.data_recon[:,self.domain_len:2*self.domain_len].reshape(end_shape))
          var3_xk = np.real(self.data_recon[:,2*self.domain_len:].reshape(end_shape))
          self.data_recon = np.moveaxis(np.array((var1_xk,var2_xk,var3_xk)),0,-1)
      elif args.dataset == 'KPP': #flatten and use POD1
          self.data_recon=np.real(self.data_recon.reshape(end_shape))
      return 1

    def save_data(self,file_str):
        print('Saving ... \t '+file_str)
        np.savez(file_str,[self.args,self.data,self.X,self.Atilde,self.Ur,self.Phi,self.Lambda,self.lv,self.b,self.args.lifts],dtype=object)
        return 1

    def load_file(self,file_str):
        print('Loading ... \t '+file_str)
        self.args,self.data,self.X,self.Atilde,self.Ur,self.Phi,self.Lambda, \
            self.lv,self.b,self.lifts = np.load(file_str,allow_pickle=True)['arr_0']
        return 1
                
    def _set_shapes(self):
        args = self.args
        if args.dataset == 'VKS':
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:-1]
            self.component_len = self.shape[-1]
        elif args.dataset == 'EE':
            self.component_len = self.shape[-1]
            self.domain_len = self.shape[1]
            self.domain_shape = [self.shape[1]]
        elif args.dataset == 'KPP':
            self.component_len = 1
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:]
        self.time_len = self.X.shape[-1]
        return 1

    def __len__(self):
      return self.time_len

    def __shape__(self):
      return self.domain_shape

    def __get_item__(self,idx):
      return None,None


class POD_DATASET(Dataset):
    def __init__(self, args):
        self.args = args
        assert args.dataset in LOADERS

        print('Loading ... \t Dataset: {}'.format(args.dataset))
        self.data_init = LOADERS[args.dataset](args.data_dir,args.eeParam)
        self.data = self.data_init.copy()
        self.shape = self.data_init.shape

        self.time_len = self.data.shape[0]
        self.data_recon=None

        if not args.load_file is None:
            self.load_file(args.load_file)
            self._set_shapes()

        args = self.args
        args.tstop = min(args.tstop,self.data_init.shape[0])

    def reduce(self):
        args = self.args
        print('Reducing ... \t Modes: {}'.format(self.args.modes))
        if args.dataset == 'VKS':
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:-1]
            self.component_len = self.shape[-1]
            self.spatial_modes, self.data, self.lv, self.ux_flux, self.uy_flux = POD2(self.data, args.tstart, args.tstop, args.modes)
        elif args.dataset == 'EE':
            self.component_len = self.shape[-1]
            self.domain_len = self.shape[1]
            self.domain_shape = [self.shape[1]]
            self.spatial_modes, self.data, self.lv, self.rho_flux, self.v_flux, self.e_flux = POD3(self.data, args.tstart, args.tstop, args.modes)
        elif args.dataset == 'KPP':
            self.component_len = 1
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:]
            self.spatial_modes, self.data, self.lv, self.h = PODKPP(self.data, args.tstart, args.tstop, args.modes)

    def reconstruct(self):
      self.data_recon = pod_mode_to_true(self,self.data,self.args)

    def save_data(self,file_str):
        args=self.args
        print('Saving ... \t '+file_str)
        if args.dataset == 'VKS':
            np.savez(file_str,[self.args,self.spatial_modes,self.data,self.lv,self.ux_flux,self.uy_flux],dtype=object)
        elif args.dataset == 'EE':
            np.savez(file_str,[self.args,self.spatial_modes,self.data,self.lv,self.rho_flux,self.v_flux,self.e_flux],dtype=object)
        elif args.dataset == 'KPP':
            np.savez(file_str,[self.args,self.spatial_modes,self.data,self.lv,self.h],dtype=object)
        return 1

    def load_file(self,file_str):
        args=self.args
        print('Loading ... \t '+file_str)
        if args.dataset == 'VKS':
            self.args,self.spatial_modes,self.data,self.lv,self.ux_flux,self.uy_flux = np.load(file_str,allow_pickle=True)['arr_0']
        elif args.dataset == 'EE':
            self.args,self.spatial_modes,self.data,self.lv,self.rho_flux,self.v_flux,self.e_flux = np.load(file_str,allow_pickle=True)['arr_0']
        elif args.dataset == 'KPP':
            self.args,self.spatial_modes,self.data,self.lv,self.h = np.load(file_str,allow_pickle=True)['arr_0']
        return 1
                
    def _set_shapes(self):
        args = self.args
        if args.dataset == 'VKS':
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:-1]
            self.component_len = self.shape[-1]
        elif args.dataset == 'EE':
            self.component_len = self.shape[-1]
            self.domain_len = self.shape[1]
            self.domain_shape = [self.shape[1]]
        elif args.dataset == 'KPP':
            self.component_len = 1
            self.domain_len = self.shape[1]*self.shape[2]
            self.domain_shape = self.shape[1:]
        return 1

    def __len__(self):
      return self.time_len

    def __shape__(self):
      return self.domain_shape

    def __get_item__(self,idx):
      return None,None

class VAE_DATASET():
    def __init__(self, args):
    
        self.pod_dataset = POD_DATASET(args)
        self.data = self.pod_dataset.data
        self.data_args = self.pod_dataset.args

        #DATA SPLITTNG
        self.train_data = self.data[:args.tr_ind, :]
        valid_data = self.data[:args.val_ind, :]

        #DATA NORMALIZATION
        self.mean_data = self.train_data.mean(axis=0)
        self.std_data = self.train_data.std(axis=0)

        train_data = (self.train_data - self.mean_data) / self.std_data
        valid_data = (valid_data - self.mean_data) / self.std_data

        #TENSOR DATA
        train_data = train_data.reshape((1, train_data.shape[0], train_data.shape[1]))
        self.train_data = torch.FloatTensor(train_data)

        valid_data = valid_data.reshape((1, valid_data.shape[0], valid_data.shape[1]))
        self.valid_data = torch.FloatTensor(valid_data)

        self.data_eval = self.data[args.val_ind:args.eval_ind, :]

        #INVERT OVER TIME
        idx = [i for i in range(self.train_data.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        self.obs_t = self.train_data.index_select(0, idx)

        #NORMALIZE TIME
        eval_times = np.linspace(0, 1, args.eval_ind)
        self.eval_times = torch.from_numpy(eval_times).float()
        self.train_times = self.eval_times[:args.tr_ind]
        self.valid_times = self.eval_times[:args.val_ind]

class SEQ_DATASET:
    def __init__(self, args):
    
        self.pod_dataset = POD_DATASET(args)
        self.data = self.pod_dataset.data
        self.data_args = self.pod_dataset.args
        if args.dataset == 'EE' and args.eeParam!=self.data_args.eeParam:
          raise Exception ('Euler Equation Params Do Not Match {} {}'.format(args.eeParam,self.data_args.eeParam))

        total_size = self.data.shape[0] - args.seq_ind
        
        #SEQUENCE DATA
        seq_data = np.vstack([[self.data[t:t + args.seq_ind, :] for t in range(total_size)]]).swapaxes(0,1)
        seq_label = np.vstack([[self.data[t+1:t+args.seq_ind+1, :] for t in range(total_size)]]).swapaxes(0,1)
        tr_ind = args.tr_ind
        val_ind = args.val_ind
        self.seq_data = seq_data
        self.seq_label = seq_label
                
        # training data
        train_data = seq_data[:, :tr_ind, :]
        train_label = seq_label[:, :tr_ind, :]
        self.train_data =  torch.FloatTensor(train_data)
        self.mean_data = train_data.reshape((-1, train_data.shape[2])).mean(axis=0)
        self.std_data = train_data.reshape((-1, train_data.shape[2])).std(axis=0)
        self.train_data = torch.FloatTensor((train_data - self.mean_data) / self.std_data).to(args.device)


        self.train_label = torch.FloatTensor(train_label)
        self.train_label = torch.FloatTensor((train_label - self.mean_data) / self.std_data).to(args.device)
        self.train_times = (torch.ones(train_data.shape[:-1])/train_data.shape[1]).to(args.device)

        # validation data
        val_data = (seq_data[:, tr_ind:val_ind, :]-self.mean_data)/self.std_data
        val_label = (seq_label[:, tr_ind:val_ind, :]-self.mean_data)/self.std_data
        self.valid_data =  torch.FloatTensor(val_data).to(args.device)
        self.valid_label = torch.FloatTensor(val_label).to(args.device)
        self.valid_times = (torch.ones(val_data.shape[:-1])/val_data.shape[1]).to(args.device)

        # validation data
        eval_data = (seq_data[:, val_ind:, :]-self.mean_data)/self.std_data
        eval_label = (seq_label[:, val_ind:, :]-self.mean_data)/self.std_data
        self.eval_data =  torch.FloatTensor(eval_data).to(args.device)
        self.eval_label = torch.FloatTensor(eval_label).to(args.device)
        self.eval_times = (torch.ones(eval_data.shape[:-1])/eval_data.shape[1]).to(args.device)



class PARAM_DATASET:

    def __init__(self, args):
        
        assert args.dataset == "EE"
        self.args = args

        print('Loading ... \t Dataset: {}'.format(args.dataset))
        self.data_init = EE_PARAM(args.data_dir)
        self.data = self.data_init
        self.num_params = self.data_init.shape[0]
        self.shape = self.data_init.shape[1:]
        self.time_len = self.shape[1]

        args.tstop = min(args.tstop, self.data.shape[1]+args.tstart-1)
        
        self.reduce()

        self.data = np.moveaxis(self.data,0,1)
        self.mean_data = self.data.mean(axis=0)
        self.std_data = self.data.std(axis=0)
        self.data = (self.data - self.mean_data) / self.std_data
        self.data = np.moveaxis(self.data,1,0) 

        #SEQUENCE DATA
        train_size = args.param_ind
        self.data_shuffled = self.data
        np.random.shuffle(self.data_shuffled)
        train =self.data_shuffled[:train_size]
        valid =self.data_shuffled[train_size:]
        train = np.moveaxis(train,0,1)
        train_data = train[:args.tr_ind]
        train_label = train[args.tr_ind:]
        valid = np.moveaxis(valid,0,1)
        valid_data = valid[:args.tr_ind]
        valid_label = valid[args.tr_ind-1:]

        train_data = torch.FloatTensor(train_data)
        train_label = torch.FloatTensor(train_label)
        valid_data = torch.FloatTensor(valid_data)
        valid_label = torch.FloatTensor(valid_label)

        padd = max(train_data.shape[0],train_label.shape[0])
        data_pad = padd-train_data.shape[0]
        label_pad = padd-train_label.shape[0]
        self.data_pad = data_pad
        self.label_pad = label_pad
        data_padding = nn.ReflectionPad1d((0,train_data.shape[0]-1))
        label_padding = nn.ReflectionPad1d((0,train_label.shape[0]-1))

        #train_data = data_padding(train_data)
        #valid_data = data_padding(valid_data)
        while train_data.shape[0]<train_label.shape[0]:
          train_data = data_padding(train_data.T).T
          valid_data = data_padding(valid_data.T).T
        while train_label.shape[0]<train_data.shape[0]:
          train_label = label_padding(train_label.T).T
          valid_label = label_padding(valid_label.T).T

        train_data=train_data[:padd]
        train_label=train_label[:padd]
        valid_data=valid_data[:padd]
        valid_label=valid_label[:padd]
        #train_data = nn.functional.pad(train_data,(0,0,0,0,0,label_pad))
        #train_label = nn.functional.pad(train_label,(0,0,0,0,0,label_pad))
        #valid_data = nn.functional.pad(valid_data,(0,0,0,0,0,data_pad))
        #valid_label = nn.functional.pad(valid_label,(0,0,0,0,0,label_pad))
        

        self.train_data = torch.FloatTensor(train_data).to(args.device)
        self.train_label = torch.FloatTensor(train_label).to(args.device)
        self.train_times = (torch.ones(train_data.shape[:-1])/train_data.shape[0]).to(args.device)

        self.valid_data =  torch.FloatTensor(valid_data).to(args.device)
        self.valid_label = torch.FloatTensor(valid_label).to(args.device)
        self.valid_times = (torch.ones(valid_data.shape[:-1])/valid_data.shape[0]).to(args.device)

    """POD Model Reduction"""
    def reduce(self):
        avg=0
        args = self.args
        Spatial_modes=[]
        Data=[]
        Lv=[]
        Rho_flux=[]
        V_flux=[]
        E_flux=[]
        for i,dat in enumerate(self.data):
            spatial_modes,temp,lv, rho_flux, v_flux, e_flux = POD3(dat, args.tstart, args.tstop, args.modes)
            Spatial_modes=Spatial_modes+[spatial_modes]
            Data=Data+[temp]
            Lv=Lv+[lv]
            Rho_flux=Rho_flux+[rho_flux]
            V_flux=V_flux+[v_flux]
            E_flux=E_flux+[e_flux]
            avg = sum(lv[:args.modes])/sum(lv) + avg
        if args.verbose: print('Avergage relative information content:',avg/self.data.shape[0])
        self.spatial_modes = np.array(Spatial_modes)
        self.data = np.array(Data)
        self.lv = np.array(Lv)
        self.rho_flux = np.array(Rho_flux)
        self.v_flux = np.array(V_flux)
        self.e_flux = np.array(E_flux)
    def reconstruct(self):
      self.data_recon = pod_mode_to_true(self,self.data,self.args)
