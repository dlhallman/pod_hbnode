import argparse

def parse_args(*sys_args):

    parser = argparse.ArgumentParser(prefix_chars='-+/',
        description='[BAYES] Bayesian parameters.')

    #DATA PARAMS
    data_parser = parser.add_argument_group('Data Parameters')
    data_parser.add_argument('--dataset', type=str, default='VKS',
                        help='Dataset types: [VKS, EE].')

    data_parser.add_argument('--data_dir', type=str, default='data/VKS.pkl',
                        help='Directory of data from cwd: sci.')

    data_parser.add_argument('--modes', type = int, default =8,
                        help = 'POD reduction modes.\nNODE model parameters.')

    data_parser.add_argument('--out_dir', type=str, default='out/VKS/BAYES',
                        help='Directory of output from cwd: sci.')

    data_parser.add_argument('--tstart', type = int, default=0,
                        help='Start time for reduction along time axis.')

    data_parser.add_argument('--tstop', type=int, default=500,
                        help='Stop time for reduction along time axis.' )
    
    data_parser.add_argument('--batch_size', type=int, default=12,
                    help='Time index for validation data.' )

    data_parser.add_argument('--seq_win', type=int, default=64,
                    help='Time index for validation data.' )

    data_parser.add_argument('--tr_win', type=int, default=150,
                    help='Time index for validation data.' )

    data_parser.add_argument('--val_win', type=int, default=200,
                    help='Time index for validation data.' )

    #MODEL PARAMS
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', type=str, default='HBNODE',
                        help='Model choices - HBNODE, NODE.')

    model_parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs.')

    model_parser.add_argument('--lr', type=float, default=.5,
                        help='Training epochs.')

    model_parser.add_argument('--factor', type=float, default = 0.975,
                        help = 'Scheduler reduction factor.')

    model_parser.add_argument('--cooldown', type=int, default = 5,
                            help = 'Scheduler cooldown factor.')

    #UNIQUE PARAMS
    uq_params = parser.add_argument_group('Unique Parameters')
    uq_params.add_argument('--verbose', type=bool, default=False,
                    help='Display full NN and all plots.')

    uq_params.add_argument('--paramEE', type=int, default=45,
                    help='Parameter index for Euler Equations.')

    #PARSE
    args, unknown = parser.parse_known_args(sys_args[0])

    return args
