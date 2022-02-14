#IMPORTS
import argparse

#PATH
import sys
sys.path.append('./')

from lib.vis.model import *
from lib.vis.modes import *

"""VISUALIZATION ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='Visualization parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--out_dir', type=str, default='./out/',
                    help='Output directory')
data_parser.add_argument('--file_list', type=str, required=True,
                    nargs='+',help='Input files sep by white space.')
data_parser.add_argument('--model_list', type=str, required=True,
                    nargs='+',help='List of models in the same order as the file list.',
                    choices=['vae_node','vae_hbnode','seq_node', 'seq_hbnode'])
data_parser.add_argument('--comparisons', type=str, required=True,
                    nargs='+',choices=['forward_nfe','backward_nfe','tr_loss','val_loss'])
data_parser.add_argument('--color_list', type=str, default=['k','r','tab:cyan','tab:green'],
                    nargs='+',choices=['forward_nfe','backward_nfe','tr_loss','val_loss'])
data_parser.add_argument('--epoch_freq', type=int, default=1,
                    help='Epoch frequency to compare models at.')
data_parser.add_argument('--verbose', default=False, action='store_true',
                help='Display full NN and all plots.')
args, unknown = parser.parse_known_args()

if 'forward_nfe' in args.comparisons:
    compare_nfe(args.file_list,args.model_list,'forward_nfe',args)   
if 'backward_nfe' in args.comparisons:
    compare_nfe(args.file_list,args.model_list,'backward_nfe',args)   
if 'tr_loss' in args.comparisons:
    compare_loss(args.file_list,args.model_list,'tr_loss',args)   
if 'val_loss' in args.comparisons:
    compare_loss(args.file_list,args.model_list,'val_loss',args)   
