#IMPORTS
import argparse
import numpy as np
from tqdm import trange

#SELF IMPORTS
import sys
sys.path.append('./')

from lib.decomp.dmd import *
from lib.datasets import * 
from lib.utils.recorder import * 
from lib.utils import *
from lib.vis.dmd import *
from lib.vis.data import *


"""MODEL ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='DMD parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE, FIB].')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/',
                    help='Directory of output from cwd: sci.')
decomp_parser = parser.add_argument_group('Decomposition Parameters')
decomp_parser.add_argument('--modes', type = int, default = 4,
                    help = 'DMD reduction modes.\nNODE model parameters.')
decomp_parser.add_argument('--tstart', type = int, default=100,
                    help='Start time for reduction along time axis.')
decomp_parser.add_argument('--tstop', type=int, default=240,
                    help='Stop time for reduction along time axis.' )
decomp_parser.add_argument('--tpred', type=int, default=300,
                    help='Prediction time.' )
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', type=bool, default=False,
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

"""INITIALIZE"""
power = args.tpred - args.tstop + 1
xr =dmd.Atilde@dmd.Ur.T@dmd.X
xk = dmd.Ur@xr
X = [xk]

"""GENERATE PREDICTIONS"""
for _ in trange(1,power, desc='DMD Generation'):

    xr =dmd.Atilde@dmd.Ur.T@xk
    xk = dmd.Ur@xr

    X = X + [xk]
X = np.array(X)

end_shape = [X.shape[0]]+list(dmd.dom_shape)
X = np.array(X).reshape(end_shape)
X =  np.moveaxis(X,1,-1)
if args.dataset == "KPP":
    X = X.T

if args.verbose: print("Generating Output ...\n",X.shape)
data_reconstruct(X,-1,args)
eig_decay(dmd,args)