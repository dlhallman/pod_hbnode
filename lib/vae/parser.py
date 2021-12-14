import argparse

def parse_args(*sys_args):

    parser = argparse.ArgumentParser(prefix_chars='-+/',
        description='[NODE] NODE parameters.')

    #DATA PARAMS
    data_parser = parser.add_argument_group('Data Parameters')
    data_parser.add_argument('--dataset', type=str, default='VKS',
                        help='Dataset types: [VKS, EE].')

    data_parser.add_argument('--data_dir', type=str, default='data/VKS.pkl',
                        help='Directory of data from cwd: sci.')

    data_parser.add_argument('--out_dir', type=str, default='out/VKS/VAE/',
                        help='Directory of output from cwd: sci.')

    data_parser.add_argument('--modes', type = int, default =8,
                        help = 'POD reduction modes.\nNODE model parameters.')

    data_parser.add_argument('--tstart', type = int, default=100,
                        help='Start time for reduction along time axis.')

    data_parser.add_argument('--tstop', type=int, default=500,
                        help='Stop time for reduction along time axis.' )
    
    data_parser.add_argument('--tr_ind', type = int, default=75,
                        help='Time index for training data.')

    data_parser.add_argument('--val_ind', type=int, default=100,
                        help='Time index for validation data.' )

    #MODEL PARAMS
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', type=str, default='NODE',
                        help='Dataset types: [NODE , HBNODE].')

    model_parser.add_argument('--epochs', type=int, default=2000,
                        help='Training epochs.')

    model_parser.add_argument('--latent_dim', type=int, default=2,
                        help = 'Size of latent dimension')

    model_parser.add_argument('--layers_enc', type=int, default=4,
                    help='Encoder Layers.')

    model_parser.add_argument('--units_enc', type=int, default=10,
                        help='Encoder units.')

    model_parser.add_argument('--layers_node', type=list, default=[12],
                    help='NODE Layers.')

    model_parser.add_argument('--units_dec', type=int, default=40,
                        help='Training iterations.')

    model_parser.add_argument('--layers_dec', type=int, default=4,
                    help='Encoder Layers.')

    model_parser.add_argument('--lr', type=float, default=0.00153,
                        help = 'Initial learning rate.')

    #UNIQUE PARAMS
    uq_params = parser.add_argument_group('Unique Parameters')
    uq_params.add_argument('--verbose', type=bool, default=False,
                    help='Display full NN and all plots.')

    uq_params.add_argument('--paramEE', type=int, default=45,
                    help='Parameter index for Euler Equations.')

    #PARSE
    args, unknown = parser.parse_known_args(sys_args[0])

    return args
