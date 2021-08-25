import os
import argparse
import json
import logging
import sys
import warnings
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.utils.ucr_datasets import load_data
from src.utils.log_utils import StreamToLogger
from src.utils.global_var import OUTPATH
from src.utils.saver import Saver
from src.utils.training_helper_sigua import main_wrapper
from src.utils.plotting_utils import plot_label_insight

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

    # Add global parameters
    parser = argparse.ArgumentParser(description='sigua single experiment')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='Plane', help='UCR datasets')

    parser.add_argument('--ni', type=float, nargs='+', default=[0, 0.15], help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80])
    parser.add_argument('--reg_term', type=float, default=1,
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--num_gradual', type=int, default=100)
    parser.add_argument('--bad_weight', type=float, default=1e-3)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=3)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


######################################################################################################
def main(args):
    # LOG STUFF
    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset))

    ## Save json of args/parameters. This is handy for TL
    # with open(os.path.join(saver.path, 'args.json'), 'w') as fp:
    #    json.dump(vars(args), fp, indent=4)

    print('run logfile at: ', os.path.join(saver.path, 'logfile.log'))
    # Logging setting
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        filename=os.path.join(saver.path, 'logfile.log'),
        filemode='a'
    )

    # Redirect stdout
    stdout_logger = logging.getLogger('STDOUT')
    slout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = slout

    # Redirect stderr
    stderr_logger = logging.getLogger('STDERR')
    slerr = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = slerr

    # Suppress output
    if args.disable_print:
        slout.terminal = open(os.devnull, 'w')
        slerr.terminal = open(os.devnull, 'w')

    ######################################################################################################
    print(args)
    print()

    ######################################################################################################
    SEED = args.seed
    # TODO: implement multi device and different GPU selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        plt.switch_backend(backend)
    print()

    ######################################################################################################
    # Data
    print('*' * shutil.get_terminal_size().columns)
    print('UCR Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)
    print()

    X, Y = load_data(args.dataset)
    x_train, x_test, Y_train_clean, Y_test_clean = train_test_split(X, Y, stratify=Y, test_size=0.2)

    Y_valid_clean = Y_test_clean.copy()
    x_valid = x_test.copy()

    batch_size = min(x_train.shape[0] // 10, args.batch_size)
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print('Batch size: ', batch_size)
    args.batch_size = batch_size
    args.test_batch_size = batch_size

    ###########################
    saver.make_log(**vars(args))

    ######################################################################################################
    df_results = main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver)

    print('Save results')
    df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)

    return df_results


######################################################################################################
if __name__ == '__main__':
    args = parse_args()
    main(args)
