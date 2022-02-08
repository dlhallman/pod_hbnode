#IMPORTS
import argparse
import numpy as np
import warnings

#SELF IMPORTS
import sys
sys.path.append('./')

from lib.decomp.pod import *
from lib.datasets import POD_DATASET 
from lib.utils.misc import set_outdir
from lib.vis.animate import data_animation
from lib.vis.modes import eig_decay, plot_mode
from lib.vis.reconstruct import data_reconstruct

#SETTINGS
warnings.filterwarnings('ignore')


"""MODEL ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='POD parameters.')
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
                    help = 'POD reduction modes.\nNODE model parameters.')
decomp_parser.add_argument('--tstart', type = int, default=0,
                    help='Start time for reduction along time axis.')
decomp_parser.add_argument('--tstop', type=int, default=399,
                    help='Stop time for reduction along time axis.' )
decomp_parser.add_argument('--tpred', type=int, default=50,
                    help='Output time for reduction along time axis.' )
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', type=bool, default=False,
                help='To display output or not.')
args, unknown = parser.parse_known_args()

assert(args.tstop-args.tstart>args.modes)

if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))
args.model ='pod'

"""FORMATTING OUT DIR"""
set_outdir(args.out_dir, args)

"""LOAD DATA"""
pod = POD_DATASET(args)
if args.load_file is None:
    pod.reduce()
    pod.save_data(args.out_dir+'/pth/'+args.dataset.lower()+'_'+str(args.tstart)+'_'+str(args.tstop)+'_pod_'+str(args.modes)+'.npz')
args = pod.args
"""RECONSTRUCTION"""
pod.reconstruct()

"""OUTPUT"""
if args.verbose: print("Generating Output ...")
eig_decay(pod,args)
data_reconstruct(pod.data_recon,args.tpred-1,args)
data_animation(pod.data_recon,args)
plot_mode(pod.data[:,:4],np.arange(args.tstart,args.tstop),args)
