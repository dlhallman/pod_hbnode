import argparse

def parse_args(*sys_args):

    parser = argparse.ArgumentParser(prefix_chars='-+/',
        description='[WALKER] WALKER parameters.')

    #DATA PARAMS
    data_parser = parser.add_argument_group('Data Parameters')
    data_parser.add_argument('--dataset', type=str, default='VKS',
                        help='Dataset types: [VKS, EE, FIB].')

    data_parser.add_argument('--data_dir', type=str, default='data/VKS.pkl',
                        help='Directory of data from cwd: sci.')

    data_parser.add_argument('--out_dir', type=str, default='out/VKS/PARAM/',
                        help='Directory of output from cwd: sci.')

    data_parser.add_argument('--modes', type = int, default = 8,
                        help = 'POD reduction modes.\nNODE model parameters.')

    data_parser.add_argument('--tstart', type = int, default=100,
                        help='Start time for reduction along time axis.')

    data_parser.add_argument('--tstop', type=int, default=500,
                        help='Stop time for reduction along time axis.' )
    
    data_parser.add_argument('--batch_size', type=int, default=20,
                    help='Time index for validation data.' )

    data_parser.add_argument('--tr_win', type=int, default=100,
                    help='Time index for validation data.' )

    #MODEL PARAMS
    model_params = parser.add_argument_group('Model Parameters')
    model_params.add_argument('--model', type=str, default='HBNODE',
                        help='Model choices - GHBNODE, HBNODE, NODE.')

    model_params.add_argument('--corr', type=int, default=0,
                        help='Skip gate input into soft max function.')

    #TRAINING PARAMS
    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--epochs', type=int, default=2000,
                        help='Training epochs.')

    train_params.add_argument('--layers', type=int, default=4,
                    help='Encoder Layers.')

    train_params.add_argument('--lr', type=float, default=0.001,
                        help = 'Initial learning rate.')

    train_params.add_argument('--factor', type=float, default=0.975,
                        help = 'Initial learning rate.')

    train_params.add_argument('--cooldown', type=int, default=5,
                        help = 'Initial learning rate.')

    train_params.add_argument('--patience', type=int, default=5,
                        help = 'Initial learning rate.')

    #UNIQUE PARAMS
    uq_params = parser.add_argument_group('Unique Parameters')
    uq_params.add_argument('--verbose', type=bool, default=False,
                    help='Display full NN and all plots.')
                    
    uq_params.add_argument('--device', type=str, default='cpu',
                    help='Device argument for training.')

    uq_params.add_argument('--paramEE', type=int, default=45,
                    help='Parameter index for Euler Equations.')

    #PARSE
    args, unknown = parser.parse_known_args(sys_args[0])

    return args
