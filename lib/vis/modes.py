#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

"""  EIGEN VALUE DECAY PLOT """
def eig_decay(dataset,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    #INITIALIZE
    total = dataset.lv.sum()
    decay=[1]
    true_switch = False #prevent overflow that occurs from dividing by total
    #CALCULATE DECAY
    for eig in dataset.lv:
        val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    x = np.arange(0,len(dataset.lv)+1)

    plt.figure(tight_layout=True)
    plt.plot(x,decay, 'k')
    #AXES
    plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
    plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
    plt.yscale('log')
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_decay').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    print("Reconstructed value is {:.5f} for {} modes.".format(1-decay[args.modes],args.modes))
    
    return 1

def plot_mode(predictions,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    for j in range(3):
        plt.figure(tight_layout=True)
        for i,node in enumerate(predictions.T):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(node, 'k')
            plt.xlabel("Time",fontsize=36)
            plt.ylabel("$\\alpha_{}$".format(i),fontsize=36)
        #OUTPUT
        end_str = str(args.dataset+'_'+args.model+'_modes').lower()
        plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
        if args.verbose: plt.show()

    return 1

def mode_prediction(predictions,true,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    for j in range(3):
        plt.figure(tight_layout=True)
        for i,node in enumerate(predictions.T):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(node, 'r', label='Prediction')
            plt.plot(true.cpu().T[i], 'k--', label='True')
            plt.xlabel("Time")
            plt.ylabel("$\\alpha_{}$".format(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
        #OUTPUT
        end_str = str(args.dataset+'_'+args.model+'_mode_prediction').lower()
        plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
        if args.verbose: plt.show()

    return 1
