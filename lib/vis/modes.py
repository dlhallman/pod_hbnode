#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

"""  EIGEN VALUE DECAY PLOT """
def eig_decay(dataset,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)

    total = dataset.lv.sum()
    decay=[1]
    #CALCULATE DECAY
    for eig in dataset.lv:
        val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    x = np.arange(0,len(decay))

    plt.figure(tight_layout=True)
    plt.plot(x,decay, 'k',linewidth=2)
    plt.xlabel('Number of Modes $(r)$',fontsize=36)
    plt.ylabel('$1-I(r)$',fontsize=36)
    plt.yscale('log')
    plt.yticks(np.logspace(-10,0,11))
    plt.ylim(1e-10,1)
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_decay').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: 
        plt.show()
        print("Relative information content is {:.5f} for {} modes.".format(1-decay[args.modes],args.modes))
    
    return 1

def plot_mode(modes,times,args):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.figure(tight_layout=True)
    for i,node in enumerate(modes.T):
        plt.subplot(2,2,i+1)
        plt.plot(times,node,'k')
        plt.xlabel("Time $(t)$",fontsize=24)
        plt.ylabel("$\\alpha_{}$".format(i),fontsize=24)
        plt.xlim(times[0],times[-1])
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_modes').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    return 1

def mode_prediction(predictions,true,times,verts,args,end_str=''):
    plt.style.use('classic')
    plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.figure(tight_layout=True)
    for i,node in enumerate(predictions.T):
        plt.subplot(2,2,i+1)
        plt.plot(times,node, 'r', dashes=[1,1], label='Prediction')
        plt.plot(times,true.T[i], 'k',dashes=[1,2], label='True')
        min_1,max_1=np.min(true.T[i]),np.max(true.T[i])
        min_2,max_2=np.min(node),np.max(node)
        min_,max_=min(min_1,min_2),max(max_1,max_2)
        plt.vlines(verts,ymin=min_+.2*min_,ymax=max_+.2*max_)
        plt.xlabel("Time $(t)$",fontsize=24)
        plt.ylabel("$\\alpha_{}$".format(i),fontsize=24)
        plt.ylim(bottom=min_+.2*min_,top=max_+.2*max_)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center',mode='expand', borderaxespad=0., prop={'size': 10}, frameon=False)
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_mode_pred'+end_str).lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    return 1
