import pandas as pd
import numpy as np
import os

#SELF IMPORTS
import sys
sys.path.append('./')
sys.path.append('../')

directory = '../data'


def FIB_DAT(out_data_dir, tws_data_dir, param=None):
   
    fileout = out_data_dir.replace("out", "full")
    fileout = fileout.replace('.dat','')
    
    # non-transient data from out_pde.dat
    data_transient = pd.read_table(out_data_dir, sep="\t", index_col=2, names=["x", "h"]).to_numpy()
    end = data_transient.shape[0]//401
    data_transient = data_transient[:,1].reshape(end,401)
    data_transient = np.delete(data_transient,400,1) # Temporary - delete last column so that it is the same shape as the tws data

    raw_data = np.loadtxt(tws_data_dir)
    vector = raw_data[:,1]
    vector = np.roll(raw_data[:,1],35)-0.7 # This contains the main traveling wave solution which is all we need
    data_nonT = vector
    
    # Create moving wave (down the rows of data_tensor)
    for i in range(31):
        vector = np.roll(vector,30)
        data_nonT = np.vstack((data_nonT,vector))
        
    data_tensor = np.vstack((data_transient,data_nonT))    
    
    np.savez(fileout, data_tensor)
    return 1


for filename1 in os.listdir(directory):
    if 'out' in filename1:
        # Found a transient dataset
        out_data_dir = os.path.join(directory, filename1)
        
        # now identify which transient dataset this is
        identifier = filename1.split('_')[2] # number associated with the L parameter
        
        # Next, find the tws dataset that corresponds to this transient data
        matches = ['tws', identifier]
        
        for filename2 in os.listdir(directory):
            if all(x in filename2 for x in matches):
                # Found the corresponding tws dataset
                tws_data_dir = os.path.join(directory, filename2)
                FIB_DAT(out_data_dir, tws_data_dir)
            







