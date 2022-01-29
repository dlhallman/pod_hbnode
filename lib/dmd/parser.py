import argparse

def parse_args(*sys_args):

    parser = argparse.ArgumentParser(prefix_chars='-+/',
        description='DMD parameters.')

    #DATA PARAMS
    data_parser = parser.add_argument_group('Data Parameters')
    data_parser.add_argument('--dataset', type=str, default='VKS',
                        help='Dataset types: [VKS, EE, FIB].')

    data_parser.add_argument('--data_dir', type=str, default='data/VKS.pkl',
                        help='Directory of data from cwd: sci.')

    data_parser.add_argument('--out_dir', type=str, default='out/VKS/DMD/',
                        help='Directory of output from cwd: sci.')

    data_parser.add_argument('--model', type = str, default = 'DMD',
                        help = 'Placeholder model name.')

    data_parser.add_argument('--modes', type = int, default = 16,
                        help = 'POD reduction modes.\nNODE model parameters.')

    data_parser.add_argument('--tstart', type = int, default=100,
                        help='Start time for reduction along time axis.')

    data_parser.add_argument('--tstop', type=int, default=240,
                        help='Stop time for reduction along time axis.' )

    data_parser.add_argument('--tpred', type=int, default=300,
                        help='Prediction time.' )


    #UNIQUE PARAMS
    uq_params = parser.add_argument_group('Unique Parameters')
    uq_params.add_argument('--verbose', type=bool, default=False,
                    help='Number of display modes.')

    uq_params.add_argument('--paramEE', type=int, default=45,
                    help='Parameter index for Euler Equations.')


    #PARSE
    args, unknown = parser.parse_known_args(sys_args[0])

    return args
