import os
import argparse
import json
import logging
import sys
import warnings
import shutil
import itertools

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
from src.utils.training_helper_v2 import *

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        self.make_log()


def plot_results(data, keys, saver, x='losses', hue='correct', col='noise', kind='box', style='whitegrid', title=None):
    sns.set_style(style)
    n = len(keys)

    for k in keys:
        g = sns.catplot(x=x, y=k, hue=hue, col=col, data=data, kind=kind)
        g.set(ylim=(0, 1))
        if title is not None:
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle('{} - {}'.format(k, title))
        saver.save_fig(g.fig, '{}_{}'.format(kind, k))


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
    parser.add_argument('--weighted', action='store_true', default=False)
    parser.add_argument('--sigma', type=float, default=0.2, help='Additive noise')

    parser.add_argument('--ni', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75], help='label noise ratio')
    parser.add_argument('--label_noise', default=0, help='Label noise type, sym or int for asymmetric, '
                                                         'number as str for time-dependent noise')

    parser.add_argument('--n_out', type=int, default=1, help='Output Heads')

    parser.add_argument('--M', type=int, nargs='+', default=[10, 30, 50, 80])
    parser.add_argument('--class_reg', type=int, default=0.88, help='Distribution regularization coeff')
    parser.add_argument('--entropy_reg', type=int, default=0, help='Entropy regularization coeff')

    parser.add_argument('--correct', nargs='+', default=[False, True], help='Correct labels')
    parser.add_argument('--track', type=int, default=5, help='Number or past predictions snapshots')
    parser.add_argument('--init_centers', type=int, default=10, help='Start phase 2')
    parser.add_argument('--correct_start', type=int, default=10, help='Start Correction')
    parser.add_argument('--correct_end', type=int, default=30,
                        help='End phase 2 - Begin phase 3 ')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--network', type=str, default='TCN',
                        help='Available networks: TCN, MLP, LSTM')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
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
    parser.add_argument('--seed', type=int, default=0, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')

    parser.add_argument('--nonlin_classifier', action='store_true', default=False, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=3)

    # TCN
    parser.add_argument('--stack', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filter_number', type=int, default=64)

    # RECURRENT
    parser.add_argument('--rnn_layers', type=int, default=2)
    # parser.add_argument('--rnn_units', type=int, default=50) # Useless for now

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[128, 128, 128])

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=True)
    parser.add_argument('--plt_recons', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=True, help='Matplotlib backend')

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
    # Main loop
    classes = len(np.unique(Y_train_clean))
    args.nbins = classes
    history = x_train.shape[1]

    # Network definition
    if args.nonlin_classifier:
        classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                      norm=args.normalization)
    else:
        classifier = LinClassifier(args.embedding_size, classes)

    if args.network == 'MLP':
        # Reshape data for MLP
        x_train = np.hstack([(x_train[:, :, i]) for i in range(x_train.shape[2])])
        x_valid = np.hstack([(x_valid[:, :, i]) for i in range(x_valid.shape[2])])
        x_test = np.hstack([(x_test[:, :, i]) for i in range(x_test.shape[2])])

        model_ae = MLPAE(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, hidden_neurons=args.neurons,
                         hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                         normalization=args.normalization).to(device)

    elif args.network == 'TCN':
        stacked_layers = RF_check(args.kernel_size, args.stack, history)

        model_ae = TCNAE(input_size=x_train.shape[2], num_filters=args.filter_number, embedding_dim=args.embedding_size,
                         seq_len=x_train.shape[1], num_stack=stacked_layers, kernel_size=args.kernel_size,
                         dropout=args.dropout, normalization=args.normalization).to(device)

    elif args.network == 'LSTM':
        # TODO: fix
        model_ae = LSTMAE(seq_len_out=x_train.shape[1], n_features=x_train.shape[2],
                          n_layers=args.rnn_layers, embedding_dim=args.embedding_size).to(device)
    else:
        raise NotImplementedError

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = AEandClass(ae=model_ae, classifier=classifier, n_out=args.n_out, name=args.network).to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % (args.network, readable(nParams))
    print(s)
    saver.append_str([s])

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Validation:', x_valid.shape, Y_valid_clean.shape,
          [(Y_valid_clean == i).sum() for i in np.unique(Y_valid_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape), 'Validation:{}'.format(x_valid.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    df_results = pd.DataFrame()
    seeds = np.random.choice(1000, args.n_runs, replace=False)
    # Creation of losses coefficient a=alpha, b=beta, g=gamma
    ag = list(itertools.product([0, 1], [0, 1]))
    abg = [[x[0], 1, x[1]] for x in ag]

    for run, seed in enumerate(seeds):
        print()
        print('#' * shutil.get_terminal_size().columns)
        # TODO: ETA and nruns
        print('EXPERIMENT: {}/{} -- RANDOM SEED:{}'.format(run + 1, args.n_runs, seed).center(columns))
        print('#' * shutil.get_terminal_size().columns)
        print()

        args.seed = seed

        # reset_seed_(seed)
        # model = reset_model(model)
        # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

        saver_seed = SaverSlave(os.path.join(saver.path, f'seed_{seed}'))
        saver_seed.append_str(['SEED: {}'.format(seed), '\r\n'])

        for coeffs in abg:
            args.abg = coeffs

            saver_coeffs = SaverSlave(os.path.join(saver_seed.path, mapper(coeffs)))
            saver_coeffs.make_log(**vars(args))
            test_results_main = collections.defaultdict(list)
            test_corrected_results_main = collections.defaultdict(list)

            print('/' * shutil.get_terminal_size().columns)
            print('Objective function: {}'.format(mapper(coeffs)))
            print('/' * shutil.get_terminal_size().columns)

            i = 0
            for ni in args.ni:
                saver_slave = SaverSlave(os.path.join(saver_coeffs.path, f'ratio_{ni}'))
                for correct_labels in args.correct:
                    # if correct_labels and args.abg[2] != 1:
                    #    # Correct label on if Lcc is active.
                    #    print('Skipping')
                    #    print(args.abg)
                    #    pass
                    # else:
                    i += 1
                    # True or false
                    print('+' * shutil.get_terminal_size().columns)
                    print('HyperRun: %d/%d' % (i, len(args.ni) * len(args.correct)))
                    print('Label noise ratio: %.3f' % ni)
                    print('Correct labels:', correct_labels)
                    print('+' * shutil.get_terminal_size().columns)

                    reset_seed_(seed)
                    model = reset_model(model)

                    Y_train, mask_train = flip_label(Y_train_clean, ni, args.label_noise)
                    Y_valid, mask_valid = flip_label(Y_valid_clean, ni, args.label_noise)
                    Y_test = Y_test_clean

                    # Re-load initial weights
                    # model.load_state_dict(torch.load(os.path.join(saver.path, 'initial_weight.pt')))

                    train_results, valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test,
                                                                                  Y_train,
                                                                                  Y_valid, Y_test, Y_train_clean,
                                                                                  Y_valid_clean,
                                                                                  mask_train, ni, args, saver_slave,
                                                                                  correct_labels,
                                                                                  plt_embedding=args.plt_embedding,
                                                                                  plt_loss_hist=args.plt_loss_hist,
                                                                                  plt_recons=args.plt_recons,
                                                                                  plt_cm=args.plt_cm,
                                                                                  weighted=args.weighted)
                    if correct_labels:
                        test_corrected_results_main = append_results_dict(test_corrected_results_main, test_results)
                    else:
                        test_results_main = append_results_dict(test_results_main, test_results)

                    test_results['noise'] = ni
                    test_results['seed'] = seed
                    test_results['correct'] = str(correct_labels)
                    test_results['losses'] = mapper(args.abg)
                    saver_coeffs.append_str(['Test Results:'])
                    saver_coeffs.append_dict(test_results)
                    df_results = df_results.append(test_results, ignore_index=True)

            plot_test_reuslts(test_results_main, test_corrected_results_main, args.ni, classes, args.network, seed,
                              saver_coeffs, abg=mapper(args.abg))

        # TODO: better way to find keys
        print('Save results')
        df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)

        try:
            keys = list(test_results_main.keys())
        except:
            keys = list(test_corrected_results_main.keys())

        # Plot structure results
        fig_title = f"Dataset:{args.dataset} - Model: {args.network} - classes:{classes} - seed:{args.seed}"
        plot_results(df_results.loc[df_results.seed == seed], keys, saver_seed, title=fig_title,
                     x='correct', hue='losses', col='noise', kind='bar', style='whitegrid')
        plot_results(df_results.loc[df_results.seed == seed], keys, saver_seed, title=fig_title,
                     x='noise', hue='losses', col='correct', kind='bar', style='whitegrid')
        # results_seed = df_results.loc[df_results.seed == seed]
        # saver_seed.append_str(['Results:', str(results_seed)])

    # End
    print('Save results')
    df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)

    # Losses column should  not change here
    fig_title = f"Dataset:{args.dataset} - Model: {args.network} - classes:{classes}"
    # plot_results(df_results, keys, saver, title=fig_title,
    #             x='losses', hue='correct', col='noise', kind='box', style='whitegrid')
    plot_results(df_results, keys, saver, title=fig_title,
                 x='correct', hue='losses', col='noise', kind='box', style='whitegrid')
    plot_results(df_results, keys, saver, title=fig_title,
                 x='noise', hue='losses', col='correct', kind='box', style='whitegrid')

    results_summary = df_results.groupby(['noise', 'correct', 'losses'])[keys].describe().T
    saver.append_str(['Results main summary', str(results_summary)])


######################################################################################################
if __name__ == '__main__':
    main()
