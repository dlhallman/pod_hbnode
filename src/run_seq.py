#IMPORTS
import argparse
import numpy as np
import time
from torchdiffeq import odeint
import torch.optim as optim
from tqdm import tqdm,trange


import sys
sys.path.append('./')

from lib.datasets import SEQ_DATASET
from lib.decomp.pod import pod_mode_to_true
from lib.models.seq import *
from lib.utils.misc import set_outdir,set_seed, Recorder
from lib.utils.seq_helper import *
from lib.vis.animate import data_animation
from lib.vis.modes import mode_prediction
from lib.vis.model import plot_loss, plot_nfe
from lib.vis.reconstruct import data_reconstruct




"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE].')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--load_file', type=str,
                    default='./out/nonT_pred/pth/vks_100_200_pod_8.npz',
                    help='Directory of pod data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/nonT_pred',
                    help='Directory of output from cwd: sci.')
data_parser.add_argument('--tr_ind', type=int, default=80,
                help='Time index for validation data.' )
data_parser.add_argument('--val_ind', type=int, default=100,
                help='Time index for validation data.' )
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--model', type=str, default='HBNODE',
                    help='Model choices - GHBNODE, HBNODE, NODE.')
data_parser.add_argument('--batch_size', type=int, default=20,
                help='Time index for validation data.' )
data_parser.add_argument('--seq_ind', type=int, default=10,
                help='Time index for validation data.' )
data_parser.add_argument('--layers', type = int, default = 8,
                    help = 'Number of hidden layers. Node: Value becomes modes*layers.')
model_params.add_argument('--corr', type=int, default=0,
                    help='Skip gate input into soft max function.')
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--epochs', type=int, default=500,
                    help='Training epochs.')
train_params.add_argument('--lr', type=float, default=0.01,
                    help = 'Initial learning rate.')
train_params.add_argument('--factor', type=float, default=0.975,
                    help = 'Initial learning rate.')
train_params.add_argument('--cooldown', type=int, default=2,
                    help = 'Initial learning rate.')
train_params.add_argument('--patience', type=int, default=5,
                    help = 'Initial learning rate.')
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--device', type=str, default='cpu',
                help='Set default torch hardware device.')
uq_params.add_argument('--seed', type=int, default=0,
                help='Set initialization seed')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='Number of display modes.')

#PARSE
args, unknown = parser.parse_known_args()
if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))


"""INITIALIZE"""
#SETTNGS
set_seed(args.seed)
#FORMAT OUTDIR
set_outdir(args.out_dir, args)
#LOAD DATA
seq = SEQ_DATASET(args)
args.modes = seq.data_args.modes


MODELS = {'NODE' : NMODEL(args),'HBNODE' : HBMODEL(args, res=True, cont=True), 'GHBNODE' : GHBMODEL(args, res=True, cont=True)}

#MODEL DIMENSIONS
assert args.model in MODELS
print('Generating ...\t Model: SEQ {}'.format(args.model))
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
                factor=args.factor, patience=args.patience, verbose=False, threshold=1e-5,
                threshold_mode='rel', cooldown=args.cooldown, min_lr=1e-7, eps=1e-08)

# TRAINING
print('Training ... \t Iterations: {}'.format(args.epochs))
epochs = trange(1,args.epochs+1)
for epoch in epochs:

    rec['epoch'] = epoch
    batchsize = args.batch_size
    train_start_time = time.time()

    #SCHEDULER
    for param_group in optimizer.param_groups:
        rec['lr'] = param_group['lr']
    scheduler.step(metrics=loss_meter_t.avg)

    #BATCHING
    for b_n in range(0, seq.train_data.shape[1], batchsize):
        model.cell.nfe = 0
        predict = model(seq.train_times[:, b_n:b_n + batchsize], seq.train_data[:, b_n:b_n + batchsize])
        loss = criteria(predict, seq.train_label[:, b_n:b_n + batchsize])
        loss_meter_t.update(loss.item())
        rec['loss'] = loss
        rec['forward_nfe'] = model.cell.nfe
        rec['forward_stiff'] = model.cell.stiff
        epochs.set_description('loss:{:.3f}'.format(loss))

        #BACKPROP
        if gradrec is not None:
            lossf = criteria(predict[-1], seq.train_label[-1, b_n:b_n + batchsize])
            lossf.backward(retain_graph=True)
            vals = model.ode_rnn.h_rnn
            for i in range(len(vals)):
                grad = vals[i].grad
                rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
            model.zero_grad()
        model.cell.nfe = 0
        loss.backward()
        optimizer.step()
        rec['backward_nfe'] = model.cell.nfe
        rec['backward_stiff'] = model.cell.stiff
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    rec['train_time'] = time.time() - train_start_time

    #VALIDATION
    if epoch == 0 or (epoch + 1) % 1 == 0:
        model.cell.nfe = 0
        predict = model(seq.valid_times, seq.valid_data)
        vloss = criteria(predict, seq.valid_label)
        rec['va_nfe'] = model.cell.nfe
        rec['va_stiff'] = model.cell.stiff
        rec['va_loss'] = vloss

    #TEST
#    if epoch == 0 or (epoch + 1) % 5 == 0:
#        model.cell.nfe = 0
#        predict = model(seq.eval_times, seq.eval_data)
#        sloss = criteria(predict, seq.eval_label)
#        sloss = sloss.detach().cpu().numpy()
#        rec['ts_nfe'] = model.cell.nfe
#        rec['ts_stiff'] = model.cell.stiff
#        rec['ts_loss'] = sloss

    #OUTPUT
    rec.capture(verbose=False)
    if (epoch + 1) % 5 == 0:
        torch.save(model, args.out_dir+'/pth/{}.mdl'.format(args.model))
        rec.writecsv(args.out_dir+'/pth/{}.csv'.format(args.model))


print("Generating Output ... ")
args.modes = seq.data_args.modes
args.model = str('seq_'+args.model).lower()
predictions = model(seq.valid_times, seq.valid_data).cpu().detach().numpy()
normalized = (predictions*seq.std_data+seq.mean_data)
times = np.arange(seq.data_args.tstart,seq.data_args.tstart+args.val_ind)
#DATA PLOTS
verts = [seq.data_args.tstart+args.tr_ind]
mode_prediction(normalized[-1,:,:4],seq.data[:args.val_ind],times,verts,args)
val_recon = pod_mode_to_true(seq.pod_dataset,normalized,args)
data_reconstruct(val_recon,args.val_ind-1,args)
data_animation(val_recon,args)

#MODEL PLOTS
plot_loss(rec_file, args)
plot_nfe(rec_file, args)
plot_AdjGrad(args.out_dir+'/pth/{}.csv'.format(args.model),args)
plot_stiff(args.out_dir+'/pth/{}.csv'.format(args.model),args)
predictions = model(seq.valid_times, seq.valid_data).cpu().detach().numpy()
val_recon = pod_mode_to_true(seq,predictions,args)
data_reconstruct(val_recon,-1,args,heat=True)
