#IMPORTS
import numpy as np
import time
from tqdm import trange
import warnings

warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('../')

#SELF IMPORTS
from sci.lib.dmd.decomp import *
from sci.lib.dmd.parser import *
from sci.lib.loader import * 
from sci.lib.recorder import * 
from sci.lib.utils import *
from sci.lib.vis.dmd import *
from sci.lib.vis.data import *


def main(parse=None):
    #ARGS INPUT
    try:
        sys_args = sys.argv[1:]
    except:
        sys_args = []
    args = parse_args(sys_args) if parse==None else parse
    if args.verbose:
        print('Parsed Arguments')
        for arg in vars(args):
            print('\t',arg, getattr(args, arg))

    #FORMAT OUTDIR
    set_outdir(args.out_dir, args)

    #DATA LOADER
    DL = DMD_LOADER(args)

    power = args.tpred - args.tstop + 1

    xr =DL.Atilde@DL.Ur.T@DL.X[-1,:]
    xk = DL.Ur@xr
    X = [xk]

    for _ in trange(1,power):

        xr =DL.Atilde@DL.Ur.T@xk
        xk = DL.Ur@xr

        X = X + [xk]
    X = np.array(X)

    end_shape = [X.shape[0]]+list(DL.dom_shape)
    X = np.array(X).reshape(end_shape)
    X =  np.moveaxis(X,1,-1)
    if args.dataset == "KPP":
        X = X.T

    print("Generating Output ... ")
    print(X.shape)
    data_reconstruct(X,-1,args)
    eig_decay(DL,args)

    return 1

if __name__ == "__main__":
    main()