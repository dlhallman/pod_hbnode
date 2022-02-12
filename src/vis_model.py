#IMPORTS
import argparse

#PATH
import sys
sys.path.append('./')

from lib.vis.animate import data_animation
from lib.vis.modes import mode_prediction
from lib.vis.model import plot_loss, plot_nfe
from lib.vis.reconstruct import data_reconstruct

"""VISUALIZATION ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='Visualization parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--out_dir', type=str, default='./out/',
                    help='Output directory')
data_parser.add_argument('--file', type=str, required=True,
                    nargs='+',help='Input files sep by white space.')
data_parser.add_argument('--model', type=str, required=True,
                    help='List of models in the same order as the file list.',
                    choices=['vae_node','vae_hbnode','seq_node', 'seq_hbnode'])
data_parser.add_argument('--epoch_freq', type=int, default=1,
                    help='Epoch frequency to compare models at.')
data_parser.add_argument('--verbose', default=False, action='store_true',
                help='Display full NN and all plots.')
args, unknown = parser.parse_known_args()


if model=='vae_node':
  from lib.utils.vae_helper import *
  enc = Encoder(latent_dim, obs_dim, args.units_enc, args.layers_enc)
  node = MODELS[args.model]
  dec = Decoder(latent_dim, obs_dim, args.units_dec, args.layers_dec)
  params = (list(enc.parameters()) + list(node.parameters()) + list(dec.parameters()))
