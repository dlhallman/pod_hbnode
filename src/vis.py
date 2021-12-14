#IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('../')

from sci.lib.vis.parser import *

#PLOT FORMATTING
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
DPI = 160

#GLOBALS
MODELS = ['NODE', 'HBNODE', 'GHBNODE']
MODELS = ['NODE', 'HBNODE']

def main():

    try:
        sys_args = sys.argv[1:]
    except:
        sys_args = []

    args = parse_args(sys_args)

    MODELS = args.models.split(',')

    dir_ = args.pth_dir

    plt.figure(tight_layout=True, dpi=DPI)
    color = ['k','r--', 'b']
    for i,model in enumerate(MODELS):
        csv_ = dir_+'/'+model+'.csv'
        with open(csv_, 'r') as file:
            df = pd.read_csv(file, index_col=False)
        loss = df['va_loss'].values
        plt.plot(loss,color[i],label=model)
        plt.yscale('log')
        # yticks = [100/(10**i) for i in range(5)]
        # plt.yticks(yticks)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    plt.savefig(dir_+'/../LOSS.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

if __name__ == "__main__":
    main()