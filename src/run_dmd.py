#IMPORTS
import argparse
import numpy as np
from tqdm import trange
import warnings

#SELF IMPORTS
import sys
sys.path.append('./')

from lib.decomp.dmd import *
from lib.datasets import DMD_DATASET 
from lib.utils.misc import set_outdir
from lib.vis.animate import data_animation
from lib.vis.modes import eig_decay
from lib.vis.reconstruct import data_reconstruct

#SETTINGS
warnings.filterwarnings('ignore')

"""MODEL ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='DMD parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE, FIB].')
data_parser.add_argument('--load_file', type=str, default=None,
                    help='Directory to load DMD data from.')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/',
                    help='Directory of output from cwd: sci.')
decomp_parser = parser.add_argument_group('Decomposition Parameters')
decomp_parser.add_argument('--modes', type = int, default = 64,
                    help = 'DMD reduction modes.\nNODE model parameters.')
decomp_parser.add_argument('--tstart', type = int, default=0,
                    help='Start time for reduction along time axis.')
decomp_parser.add_argument('--tstop', type=int, default=101,
                    help='Stop time for reduction along time axis.' )
decomp_parser.add_argument('--tpred', type=int, default=400,
                    help='Prediction time.' )
decomp_parser.add_argument('--lifts', type=str, default='', nargs='+',
                    help='Prediction time.' )
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='To display output or not.')
args, unknown = parser.parse_known_args()

if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))
args.model ='dmd'

"""FORMATTING OUT DIR"""
set_outdir(args.out_dir, args)

"""LOAD DATA"""
dmd = DMD_DATASET(args)
if args.load_file is None:
    dmd.reduce()
    dmd.save_data(args.out_dir+'/pth/'+args.dataset.lower()+'_'+str(args.tstart)+'_'+str(args.tstop)+'_dmd_'+str(args.modes)+'.npz')
args = dmd.args

"""INITIALIZE"""
Xk = np.array(dmd.X.T[0][:dmd.domain_len*dmd.component_len])

"""GENERATE PREDICTIONS"""
for k in trange(1,args.tpred, desc='DMD Generation'):
    Lambda_k = np.linalg.matrix_power(dmd.Lambda,k)
    xk=(dmd.Phi@Lambda_k@dmd.b)[:dmd.domain_len*dmd.component_len]
    Xk=np.vstack((Xk,xk))

"""RECONSTRUCTION"""
Xk = np.array(Xk)
dmd.data_recon = Xk
dmd.reconstruct()

"""OUTPUT"""
if args.verbose: print("Generating Output ...\n",Xk.shape)
eig_decay(dmd,args)
data_reconstruct(dmd.data_recon,args.tpred-1,args)
data_animation(dmd.data_recon,args)
