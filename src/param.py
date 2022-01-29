#IMPORTS
import time
import torch
import torch.nn as nn
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('../')

#SELF IMPORTS
from sci.lib.loader import * 
from sci.lib.param.models import *
from sci.lib.param.parser import *
from sci.lib.recorder import * 
from sci.lib.utils import *
from sci.lib.vis.param import *
from sci.lib.vis.universal import *


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

    MODELS = {'NODE' : NMODEL(args),'HBNODE' : HBMODEL(args, res=True, cont=True), 'GHBNODE' : GHBMODEL(args, res=True, cont=True)}
    #FORMAT OUTDIR
    set_outdir(args.out_dir, args)

    #DATA LOADER
    DL = PARAM_LOADER(args)

    
    #MODEL DIMENSIONS
    assert args.model in MODELS
    print('Generating ...\t Model: WALKER {}'.format(args.model))
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
    for epoch in trange(args.epochs):

        rec['epoch'] = epoch
        batchsize = args.batch_size
        train_start_time = time.time()

        #SCHEDULER
        for param_group in optimizer.param_groups:
            rec['lr'] = param_group['lr']
        scheduler.step(metrics=loss_meter_t.avg)

        #BATCHING
        for b_n in range(0, DL.train_data.shape[1], batchsize):
            model.cell.nfe = 0
            predict = model(DL.train_times[:, b_n:b_n + batchsize], DL.train_data[:, b_n:b_n + batchsize])
            loss = criteria(predict, DL.train_label[:, b_n:b_n + batchsize])
            loss_meter_t.update(loss.item())
            rec['loss'] = loss

            #BACKPROP
            if gradrec is not None:
                lossf = criteria(predict[-1], DL.train_label[-1, b_n:b_n + batchsize])
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
            predict = model(DL.eval_times, DL.eval_data)
            sloss = criteria(predict, DL.eval_label)
            sloss = sloss.detach().cpu().numpy()
            rec['ts_nfe'] = model.cell.nfe
            rec['ts_loss'] = sloss

        #OUTPUT
        rec.capture(verbose=False)
        if (epoch + 1) % 5 == 0:
            torch.save(model, args.out_dir+'/pth/{}.mdl'.format(args.model))
            rec.writecsv(args.out_dir+'/pth/{}.csv'.format(args.model))
    print("Generating Output ... ")
    param_Loss(args.out_dir+'/pth/{}.csv'.format(args.model),args)
    param_ModesLong(DL, model, args)
    
if __name__ == "__main__":
    main()