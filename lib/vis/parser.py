import argparse

def parse_args(*sys_args):

    parser = argparse.ArgumentParser(prefix_chars='-+/',
        description='[WALKER] WALKER parameters.')

    #DATA PARAMS
    data_parser = parser.add_argument_group('Data Parameters')

    data_parser.add_argument('--pth_dir', type=str, default='out/VKS/SEQ/pth/',
                        help='Pointer to pth directory from cwd: sci.')

    data_parser.add_argument('--models', type=str, default='NODE,HBNODE,GHBNODE',
                        help='Comma separated model options.')

    data_parser.add_argument('--verbose', type=bool, default=False,
                        help='Comma separated model options.')

    #PARSE
    args, unknown = parser.parse_known_args(sys_args[0])

    return args
