from itertools import product
import random

import sys
sys.path.append('../')

import sci.lib.vae.parser as nparse
import sci.lib.hbnode.parser as hbparse
import sci.src.node as node
import sci.src.hbnode as hbnode

"""
Variables:
    Latent Dimension : {2,5,7}
    Encoder Layers : {1,3,5}
    Encoder Units : { 10,20,40}
    Node Layers : {[12], [6,6], [12,12], [24]}
    Decoder Units : {10,20,40}
    Decoder Layers : {1,3,5}
    Learning Rate : {.005, .0025, .001}
"""

LDIM = [2,4]
ELAYER = [2,5]
EUNIT = [10,20,40]
NLAYER = [[12],[6,6],[12,12]]
DUNIT = [10,20,40]
DLAYER = [2,5]
LR = [.005, .0025, .001]

#PRODUCT ARGS 
PRODS = list(product(LDIM,ELAYER, EUNIT, NLAYER, DUNIT,DLAYER, LR))

def main(*sys_args):

    for i in range(len(PRODS[:100])):
        #NODE PARAMS
        args1 = nparse.parse_args([])
        args2 = hbparse.parse_args([])
        args1.epochs=1000
        args1.latent_dim = PRODS[i][0]
        args1.layers_enc = PRODS[i][1]
        args1.units_enc = PRODS[i][2]
        args1.layers_node = PRODS[i][3]
        args1.units_dec = PRODS[i][4]
        args1.layers_dec = PRODS[i][5]
        args1.lr = PRODS[i][6]
        args1.out_dir = 'out/grid_search/node_'+str(i)       

        #HBNODE PARAMS
        args2.epochs=1000
        args2.latent_dim = PRODS[i][0]
        args2.layers_enc = PRODS[i][1]
        args2.units_enc = PRODS[i][2]
        args2.layers_node = PRODS[i][3]
        args2.units_dec = PRODS[i][4]
        args2.layers_dec = PRODS[i][5]
        args2.lr = PRODS[i][6] 
        args2.out_dir = 'out/grid_search/hbnode_'+str(i)
        print('Grid Search: NODE {}'.format(i))
        node.main(args1)
        print('Grid Search: HBNODE {}'.format(i))
        hbnode.main(args2)

if __name__ == "__main__":
    print('Computing for {} iterations'.format(len(PRODS)))
    main()