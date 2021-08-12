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
from tslearn.datasets import UCR_UEA_datasets

from src.utils.ucr_datasets import load_data
from src.utils.log_utils import StreamToLogger
from src.utils.new_dataload import OUTPATH
from src.utils.saver import Saver
from src.utils.training_helper_v2 import main_wrapper, plot_label_insight

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
    parser = argparse.ArgumentParser(description='Toy Classification')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')
    parser.add_argument('--data_split', type=str, default='random20', choices=['original', 'random20'],
                        help='train-test splitting strategy')
    parser.add_argument('--weighted', action='store_true', default=True)
    parser.add_argument('--sigma', type=float, default=0.0, help='Additive noise')

    parser.add_argument('--ni', type=float, nargs='+', default=[0.30], help='label noise ratio')
    parser.add_argument('--label_noise', default=0, help='Label noise type, sym or int for asymmetric, '
                                                         'number as str for time-dependent noise')

    parser.add_argument('--n_out', type=int, default=1, help='Output Heads')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40 ,60, 80])
    parser.add_argument('--abg', type=float, nargs='+', default=[1, 1, 1])  # AE - Classification - Cluster
    parser.add_argument('--class_reg', type=int, default=1, help='Distribution regularization coeff')
    parser.add_argument('--entropy_reg', type=int, default=0., help='Entropy regularization coeff')

    parser.add_argument('--correct', nargs='+', default=[True], help='Correct labels')
    parser.add_argument('--track', type=int, default=5, help='Number or past predictions snapshots')
    parser.add_argument('--init_centers', type=int, default=1, help='Start phase 2')
    parser.add_argument('--correct_start', type=int, default=26, help='Start Correction')
    parser.add_argument('--correct_end', type=int, default=56,
                        help='End phase 2 - Begin phase 3 ')
    #TODO: polish code
    parser.add_argument('--delta_start', type=int, default=20,
                        help='End phase 2 - Begin phase 3 ')
    parser.add_argument('--delta_end', type=int, default=20,
                        help='End phase 2 - Begin phase 3 ')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--network', type=str, default='CNN',
                        help='Available networks: TCN, MLP, LSTM, CNN')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--patience_stopping', type=int, default=100000)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--optimizer', type=str, default='torch.optim.Adam')
    parser.add_argument('--class_loss', type=str, default='CrossEntropy',
                        choices=['CrossEntropy', 'Taylor', 'GeneralizedCE', 'Unhinged', 'PHuber', 'PHuberGeneralized'])

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--metrics', nargs='+',  # TODO
                        default=('acc', 'prec', 'rec'))

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=1, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')

    parser.add_argument('--nonlin_classifier', action='store_true', default=True, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=3)

    # TCN
    parser.add_argument('--stack', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filter_number', type=int, default=64)

    # CNN
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # RECURRENT
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_units', type=int, default=50)

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[128, 128, 128])

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_loss', action='store_true', default=True)
    parser.add_argument('--plt_embedding', action='store_true', default=True)
    parser.add_argument('--plt_loss_hist', action='store_true', default=True)
    parser.add_argument('--plt_cm', action='store_true', default=True)
    parser.add_argument('--plt_recons', action='store_true', default=True)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


######################################################################################################
def main():
    args = parse_args()
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
    # LOG STUFF
    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  network=os.path.join(args.dataset, args.network))

    # Save json of args/parameters. This is handy for TL
    with open(os.path.join(saver.path, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

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
    # Data
    print('*' * shutil.get_terminal_size().columns)
    print('UCR Dataset: {} - split-mode: {}'.format(args.dataset, args.data_split).center(columns))
    print('*' * shutil.get_terminal_size().columns)
    print()

    if args.data_split == 'original':
        x_train, Y_train_clean, x_test, Y_test_clean = load_data(args.dataset, data_split=args.data_split)
    else:
        X, Y = load_data(args.dataset, data_split=args.data_split)
        x_train, x_test, Y_train_clean, Y_test_clean = train_test_split(X, Y, stratify=Y, test_size=0.2)

    # # Add noise
    # x_train += np.random.normal(0, args.sigma, x_train.shape)
    # x_test += np.random.normal(0, args.sigma, x_test.shape)

    Y_valid_clean = Y_test_clean.copy()
    x_valid = x_test.copy()

    history = x_train.shape[1]
    batch_size = min(x_train.shape[0] // 10, args.batch_size)
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print('Batch size: ', batch_size)
    args.batch_size = batch_size

    args.weights = None
    if args.weighted:
        print('Weighted')
        nSamples = np.unique(Y_train_clean, return_counts=True)[1]
        tot_samples = len(Y_train_clean)
        weights = (nSamples / tot_samples).max() / (nSamples / tot_samples)
        args.weights = weights

    ###########################
    saver.make_log(**vars(args))
    plot_label_insight(x_train, Y_train_clean, saver=saver)

    ######################################################################################################
    df_results = main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver)

    print('Save results')
    df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)

######################################################################################################
if __name__ == '__main__':
    main()
