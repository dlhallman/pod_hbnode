#IMPORTS
import argparse
import time
import torch
import torch.nn as nn
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('./')

#SELF IMPORTS
from lib.datasets import * 
from lib.models.param import *
from lib.utils.misc import *
from lib.vis.model import *

parser = argparse.ArgumentParser(prefix_chars='-+/',
                  description='[PARAMTERIZED] PARAMETERIZED parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='EE',
                  help='Dataset types: [EE].')
data_parser.add_argument('--data_dir', type=str, default='data/EulerEqs.npz',
                  help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/ee/',
                  help='Directory of output from cwd: sci.')
data_parser.add_argument('--modes', type = int, default = 4,
                  help = 'POD reduction modes.')
data_parser.add_argument('--tstart', type = int, default=0,
                  help='Start time for reduction along time axis.')
data_parser.add_argument('--tstop', type=int, default=181,
                  help='Stop time for reduction along time axis.' )
data_parser.add_argument('--batch_size', type=int, default=80,
              help='Time index for validation data.' )
data_parser.add_argument('--tr_ind', type=int, default=100,
              help='Time index for validation data.' )
data_parser.add_argument('--seq_ind', type=int, default=100,
              help='Time index for validation data.' )
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--model', type=str, default='NODE',
                  help='Model choices - GHBNODE, HBNODE, NODE.')
model_params.add_argument('--corr', type=int, default=0,
                  help='Skip gate input into soft max function.')
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--epochs', type=int, default=500,
                  help='Training epochs.')
train_params.add_argument('--layers', type=int, default=1,
              help='Encoder Layers.')
train_params.add_argument('--lr', type=float, default=0.01,
                  help = 'Initial learning rate.')
train_params.add_argument('--factor', type=float, default=0.975,
                  help = 'Initial learning rate.')
train_params.add_argument('--cooldown', type=int, default=0,
                  help = 'Initial learning rate.')
train_params.add_argument('--patience', type=int, default=5,
                  help = 'Initial learning rate.')
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', type=bool, default=False,
              help='Display full NN and all plots.')
uq_params.add_argument('--device', type=str, default='cpu',
              help='Device argument for training.')

args, unknown = parser.parse_known_args()


if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))

"""INITIALIZE"""
#FORMAT OUTDIR
set_outdir(args.out_dir, args)
#DATA LOADER
param = PARAM_DATASET(args)

MODELS = {'NODE' : NMODEL(args),'HBNODE' : HBMODEL(args, res=True, cont=True), 'GHBNODE' : GHBMODEL(args, res=True, cont=True)}

#MODEL DIMENSIONS
assert args.model in MODELS
print('Generating ...\t Model: PARAMETER {}'.format(args.model))
model = MODELS[args.model].to(args.device)
if args.verbose:
    print(model.__str__())
    print('Number of Parameters: {}'.format(count_parameters(model)))

#LEARNING UTILITIES
gradrec = True
torch.manual_seed(0)
rec = Recorder()
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
loss_meter_t = RunningAverageMeter()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=args.factor, patience=args.patience, verbose=args.verbose, threshold=1e-5,
                threshold_mode='rel', cooldown=args.cooldown, min_lr=1e-7, eps=1e-08)

print("Training ...")
# TRAINING
epochs = trange(args.epochs)
for epoch in epochs:

    rec['epoch'] = epoch
    batchsize = args.batch_size
    train_start_time = time.time()

    #SCHEDULER
    for param_group in optimizer.param_groups:
        rec['lr'] = param_group['lr']
    scheduler.step(metrics=loss_meter_t.avg)

    #BATCHING
    for b_n in range(0, param.train_data.shape[1], batchsize):
        model.cell.nfe = 0
        predict = model(param.train_times[:, b_n:b_n + batchsize], param.train_data[:, b_n:b_n + batchsize,:])
        loss = criteria(predict, param.train_label[:,b_n:b_n + batchsize,:])
        loss_meter_t.update(loss.item())
        rec['loss'] = loss
        epochs.set_description('loss:{:.3f}'.format(loss))

        #BACKPROP
        if gradrec is not None:
            lossf = criteria(predict[-1], param.train_label[-1, b_n:b_n + batchsize])
            lossf.backward(retain_graph=True)
            vals = model.ode_rnn.h_rnn
            for i in range(len(vals)):
                grad = vals[i].grad
                rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
            model.zero_grad()
        model.cell.nfe = 0
        loss.backward()
        rec['backward_nfe'] = model.cell.nfe
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    rec['train_time'] = time.time() - train_start_time

    #TEST
    if epoch == 0 or (epoch + 1) % 1 == 0:
        model.cell.nfe = 0
        predict = model(param.valid_times, param.valid_data)
        sloss = criteria(predict, param.valid_label)
        sloss = sloss.detach().cpu().numpy()
        rec['ts_nfe'] = model.cell.nfe
        rec['ts_loss'] = sloss

    #OUTPUT
    rec.capture(verbose=False)
    if (epoch + 1) % 5 == 0:
        torch.save(model, args.out_dir+'/pth/{}.mdl'.format(args.model))
        rec.writecsv(args.out_dir+'/pth/{}.csv'.format(args.model))
print("Generating Output ... ")
rec_file = args.out_dir+ './pth/'+args.model+'.csv'
rec.writecsv(rec_file)
args.model = str('param_'+args.model).lower()
tr_pred= model(param.train_times, param.train_data).cpu().detach().numpy()[-1]
val_pred = model(param.valid_times, param.valid_data).cpu().detach().numpy()[-1]
predictions=np.vstack((tr_pred,val_pred))
normalized = (predictions*param.std_data+param.mean_data)
print(normalized.shape)
times = np.arange(args.tstart+args.seq_ind,args.tstop)
#DATA PLOTS
#verts = [args.tstart+args.tr_ind]
#mode_prediction(normalized[:,:4],param.param[-1,:args.val_ind],times,verts,args)
#val_recon = pod_mode_to_true(param.pod_dataset,normalized,args)
#data_reconstruct(val_recon,-1,args)
#data_animation(val_recon,args)

#MODEL PLOTS
#plot_loss(rec_file, args)
#plot_nfe(rec_file,'forward_nfe', args)
#plot_adjGrad(rec_file, args)
#plot_stiff(rec_file, args)
