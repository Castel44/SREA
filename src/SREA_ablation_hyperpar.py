import argparse
import copy
import itertools
import os
import shutil
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from src.utils.SREA_utils import single_experiment_SREA
from src.utils.global_var import OUTPATH
from src.utils.saver import Saver
from src.utils.utils import str2bool, map_abg_main, map_losstype, check_ziplen, remove_duplicates

######################################################################################################

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


######################################################################################################
def run(x, args, path, result_path, id):
    print('Process PID:', os.getpid())
    track, init_centers, delta_start, delta_end, abg = x
    args.abg = abg
    args.track = track
    args.init_centers = init_centers
    args.correct_start = init_centers + delta_start
    args.correct_end = init_centers + delta_start + delta_end
    args.delta_start = delta_start
    args.delta_end = delta_end
    args.id = id

    df_run = single_experiment_SREA(args, path)
    if os.path.exists(result_path):
        df_run.to_csv(result_path, mode='a', sep=',', header=False, index=False)
    else:
        df_run.to_csv(result_path, mode='a', sep=',', header=True, index=False)


def parse_args():
    """
    Parse arguments
    """
    # Add global parameters
    parser = argparse.ArgumentParser(
        description='SREA hyperparameter ablation with UCR datasets.'
                    'Each hyperpar is passed as a list a a grid with all the combination is exploited.'
                    'Parallel implementation on single GPU.')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')

    parser.add_argument('--ni', type=float, nargs='+', default=[0.2, 0.6], help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80])
    parser.add_argument('--abg', type=float, nargs='+', default=None,
                        help='Loss function coefficients. a (alpha) = AE, b (beta) = classifier, g (gamma) = clusterer'
                             'Set to None to iterate on all combination of a/b/g.')
    parser.add_argument('--class_reg', type=int, default=1, help='Distribution regularization coeff')
    parser.add_argument('--entropy_reg', type=int, default=0, help='Entropy regularization coeff')

    # Hyperpar
    parser.add_argument('--correct', nargs='+', type=str2bool, default=[True], help='Correct labels')
    parser.add_argument('--delta_track', nargs='+', type=int, default=[5])
    parser.add_argument('--init', nargs='+', type=int, default=[1])
    parser.add_argument('--delta_init', nargs='+', type=int, default=[0])
    parser.add_argument('--delta_end', nargs='+', type=int, default=[30])

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--patience_stopping', type=int, default=100000)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=0, help='RNG seed.')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--process', type=int, default=2, help='Number of parallel process. Single GPU.')

    parser.add_argument('--nonlin_classifier', action='store_true', default=True, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=3)

    # CNN
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=True)
    parser.add_argument('--plt_loss', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--plt_recons', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=True, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


######################################################################################################
def main():
    args = parse_args()
    assert args.correct.__len__() <= 2, "'correct' argument must be True, False or Both. Now is: {}".format(
        args.correct)
    print(args)
    print()

    ######################################################################################################
    SEED = args.init_seed
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
                  hierarchy=os.path.join(args.dataset, map_losstype(args.label_noise),
                                         map_abg_main(args.abg)))
    saver.make_log(**vars(args))

    ######################################################################################################
    # Main loop
    csv_path = os.path.join(saver.path, '{}_results.csv'.format(args.dataset))

    # Create alpha beta gamma hyperparameter
    if args.abg == None:
        ag = list(itertools.product([0, 1], [0, 1]))
        args.abg = [[x[0], 1, x[1]] for x in ag]
    else:
        args.abg = [args.abg]

    # Creation of losses coefficient a=alpha, b=beta, g=gamma
    hyper_list = list(itertools.product(args.delta_track, args.init, args.delta_init, args.delta_end, args.abg))
    hyper_run = pd.DataFrame(hyper_list, columns=['track', 'init_centers', 'delta_start', 'delta end', 'losses'])
    total_run = len(hyper_run)
    hyper_list = check_ziplen(hyper_list, args.process)

    saver.append_str(['\r\n', 'Hyperparameters Combinations:', str(hyper_run)])

    total_optim_loop = total_run * args.n_runs * len(args.ni) * len(args.correct)
    print(f'Hyperpar combinations: {total_run}')
    print(f'Total training call: {total_optim_loop}')
    saver.append_str(['Total Run:', total_optim_loop])

    iterator = zip(*[hyper_list[j::args.process] for j in range(args.process)])
    total_iter = len(list(iterator))

    torch.multiprocessing.set_start_method('spawn', force=True)
    start_datetime = datetime.now()
    print('{} - Start Experiments. {} parallel process. Single GPU. \r\n'.format(
        start_datetime.strftime("%d/%m/%Y %H:%M:%S"), args.process))
    for i, (x) in enumerate(zip(*[hyper_list[j::args.process] for j in range(args.process)])):
        start_time = time.time()
        x = remove_duplicates(x)
        n_process = len(x)
        idxs = [i * args.process + j for j in range(1, n_process + 1)]
        print('/' * shutil.get_terminal_size().columns)
        print(
            'ITERATION: {}/{}'.format(idxs, total_run).center(columns))
        print('/' * shutil.get_terminal_size().columns)

        process = []
        print('Hyperparameters:')
        for j, id in zip(range(n_process), idxs):
            process.append(mp.Process(target=run, args=(x[j], copy.deepcopy(args), saver.path, csv_path, id)))
            print(hyper_run.iloc[id - 1].to_string())
            print()

        for p in process:
            p.start()

        for p in process:
            p.join()

        end_time = time.time()
        iter_seconds = end_time - start_time
        total_seconds = end_time - start_datetime.timestamp()
        print('Iteration time: {} - ETA: {}'.format(time.strftime("%Mm:%Ss", time.gmtime(iter_seconds)),
                                                    time.strftime('%Hh:%Mm:%Ss',
                                                                  time.gmtime(
                                                                      total_seconds * (total_iter / (i + 1) - 1)))))
        print()

    print('*' * shutil.get_terminal_size().columns)
    print('DONE!')
    end_datetime = datetime.now()
    total_seconds = (end_datetime - start_datetime).total_seconds()
    print('{} - Experiment took: {}'.format(end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                                            time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_seconds))))
    print(f'results dataframe saved in: {csv_path}')


######################################################################################################
if __name__ == '__main__':
    main()
