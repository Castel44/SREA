import argparse
import copy
import os
import shutil
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.multiprocessing as mp

from BMM_single_experiment import main as main_BMM
from coteaching_single_experiment import main as main_coteaching
from sigua_single_experiment import main as main_sigua
from src.utils.global_var import OUTPATH
from src.utils.plotting_utils import plot_results
from src.utils.saver import Saver

######################################################################################################
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='BMM - Sigua - Coteaching experiments with UCR datasets.'
                                                 'Parallel run of each algorithm on single GPU.')

    parser.add_argument('--dataset', type=str, default='Plane', help='UCR datasets')
    parser.add_argument('--ni', type=float, nargs='+', default=[0, 0.3, 0.6], help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80])
    parser.add_argument('--reg_term', type=float, default=1,
                        help="Parameter of the regularization term, default: 0.")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')

    # Only SIGUA AND COTEACHING
    parser.add_argument('--num_gradual', type=int, default=100)
    parser.add_argument('--bad_weight', type=float, default=1e-3)

    # Only BMM
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--Mixup', type=str, default='Dynamic', choices=['None', 'Static', 'Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--correct', type=str, nargs='+', default=['none', 'Mixup', 'MixUp-BMM'], help='Correct labels')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')

    parser.add_argument('--optimizer', type=str, default='torch.optim.Adam')
    parser.add_argument('--class_loss', type=str, default='CrossEntropy',
                        choices=['CrossEntropy', 'Taylor', 'GeneralizedCE', 'Unhinged', 'PHuber', 'PHuberGeneralized'])

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=32)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=True)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=True, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


def run(args, main_func, result_path):
    print('Process PID:', os.getpid())

    df_run = main_func(args)
    if os.path.exists(result_path):
        df_run.to_csv(result_path, mode='a', sep=',', header=False, index=False)
    else:
        df_run.to_csv(result_path, mode='a', sep=',', header=True, index=False)


if __name__ == '__main__':
    args = parse_args()
    start_datetime = datetime.now()
    print('{} - Start Experiments \r\n'.format(
        start_datetime.strftime("%d/%m/%Y %H:%M:%S")))

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        plt.switch_backend(backend)
    print()

    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset))
    saver.make_log(**vars(args))
    csv_path = os.path.join(saver.path, '{}_results.csv'.format(args.dataset))

    args_BMM = copy.deepcopy(args)
    vars(args_BMM)['ni'] = [x * 100 for x in vars(args_BMM)['ni']]

    torch.multiprocessing.set_start_method('spawn', force=True)

    process = []
    process.append(mp.Process(target=run, args=(args_BMM, main_BMM, csv_path)))
    process.append(mp.Process(target=run, args=(args, main_sigua, csv_path)))
    process.append(mp.Process(target=run, args=(args, main_coteaching, csv_path)))

    for p in process:
        p.start()
        time.sleep(1)

    for p in process:
        p.join()

    print('*' * shutil.get_terminal_size().columns)
    print('DONE!')
    end_datetime = datetime.now()
    total_seconds = (end_datetime - start_datetime).total_seconds()
    print('{} - Experiment took: {}'.format(end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                                            time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_seconds))))
    print(f'results dataframe saved in: {csv_path}')

    ####
    df = pd.read_csv(csv_path)
    title = "Dataset: {}".format(args.dataset)
    plot_results(df, ['acc', 'f1_weighted'], saver, x='noise', hue='correct', title=title)
