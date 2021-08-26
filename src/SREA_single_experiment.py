import argparse
import logging
import os
import shutil
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.utils.SREA_utils import main_wrapper
from src.utils.global_var import OUTPATH
from src.utils.log_utils import StreamToLogger
from src.utils.plotting_utils import plot_label_insight
from src.utils.saver import Saver
from src.utils.ucr_datasets import load_data

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

def parse_args():
    # Add global parameters
    parser = argparse.ArgumentParser(
        description='SREA Single Experiment. It run n_runs independent experiments with different random seeds.'
                    ' Each run evaluate different noise ratios (ni).')

    parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')

    parser.add_argument('--ni', type=float, nargs='+', default=[0, 0.30], help='label noise ratio')
    parser.add_argument('--label_noise', default=0, help='Label noise type, sym or int for asymmetric, '
                                                         'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80], help='Scheduler milestones')
    parser.add_argument('--abg', type=float, nargs='+',
                        help='Loss function coefficients. a (alpha) = AE, b (beta) = classifier, g (gamma) = clusterer',
                        default=[1, 1, 1])
    parser.add_argument('--class_reg', type=int, default=1, help='Distribution regularization coeff')
    parser.add_argument('--entropy_reg', type=int, default=0., help='Entropy regularization coeff')

    parser.add_argument('--correct', nargs='+', default=[True],
                        help='Correct labels. Set to false to not correct labels.')
    parser.add_argument('--track', type=int, default=5, help='Number or past predictions snapshots')
    parser.add_argument('--init_centers', type=int, default=1, help='Initialize cluster centers. Warm up phase.')
    parser.add_argument('--delta_start', type=int, default=10, help='Start re-labeling phase')
    parser.add_argument('--delta_end', type=int, default=30,
                        help='Begin fine-tuning phase')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=1, help='Initial RNG seed. Only for reproducibility')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs, each run has different rng seed.')

    parser.add_argument('--classifier_dim', type=int, default=32, help='Dimension of final classifier')
    parser.add_argument('--embedding_size', type=int, default=3, help='Dimension of embedding')

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False,
                        help='Suppress screen print, keep log.txt')
    parser.add_argument('--plt_loss', action='store_true', default=True, help='plot loss function each epoch')
    parser.add_argument('--plt_embedding', action='store_true', default=True, help='plot embedding representation')
    parser.add_argument('--plt_loss_hist', action='store_true', default=True,
                        help='plot loss history for clean and mislabled samples')
    parser.add_argument('--plt_cm', action='store_true', default=True, help='plot confusion matrix')
    parser.add_argument('--plt_recons', action='store_true', default=True, help='plot AE reconstructions')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Matplotlib backend. Set true if no display connected.')

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
                  hierarchy=os.path.join(args.dataset))

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

    # Suppress terminal output
    if args.disable_print:
        slout.terminal = open(os.devnull, 'w')
        slerr.terminal = open(os.devnull, 'w')

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
