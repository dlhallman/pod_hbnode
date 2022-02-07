#IMPORTS
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('../')

"""VISUALIZATION ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='Visualization parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE, FIB].')
data_parser.add_argument('--model', type=str, default='POD',
                    help='Dataset types: [POD, DMD].')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/',
                    help='Directory of output from cwd: sci.')
data_parser.add_argument('--modes', type = int, default = 8,
                    help = 'Decomposition modes.')
args, unknown = parser.parse_known_args()

if args.model in ('POD','DMD'):
    data = npz.load
