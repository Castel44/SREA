import collections
import os
import shutil
import matplotlib as mpl

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


from src.models.model import CNNAE
from src.models.MultiTaskClassification import MetaModel, LinClassifier, NonLinClassifier
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class
from src.utils.plotting_utils import plot_results, plot_embedding

######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns

######################################################################################################
def sigua_loss(model_loss, rt, bad_weight):
    num_total_data = int(model_loss.size(0))
    num_good_data = int(num_total_data * rt)
    num_bad_data = int((num_total_data - num_good_data) / 2)  # TODO: Bad data ratio should be changed

    _, good_data_idx = torch.topk(model_loss, k=num_good_data, largest=False)  # small loss samples
    _, good_and_bad_data_idx = torch.topk(model_loss, k=num_good_data + num_bad_data,
                                          largest=False)  # small loss samples

    bad_data_idx = good_and_bad_data_idx[num_good_data:]

    # co-teaching
    model_loss_filter = torch.zeros((model_loss.size(0))).cuda()
    model_loss_filter[good_data_idx] = 1.0  # good data
    model_loss_filter[bad_data_idx] = -1.0 * bad_weight  # bad data

    model_loss = (model_loss_filter * model_loss).mean()

    return model_loss


def train_step(data_loader, model, optimizer, criterion, rt, bad_weight=0.001):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model = model.train()
    for x, y_hat in data_loader:
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        out = model(x)

        model_loss = criterion(out, y_hat)
        model_loss = sigua_loss(model_loss=model_loss, rt=rt, bad_weight=bad_weight)

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, model


def test_step(data_loader, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def valid_step(data_loader, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)


def train_model(model, train_loader, valid_loader, test_loader, args, tau):
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    # learning history
    train_acc_list = []
    test_acc_list = []
    try:
        for e in range(args.epochs):
            # update reduce step
            rt = update_reduce_step(cur_step=e, num_gradual=args.num_gradual, tau=tau)

            # training step
            train_accuracy, avg_loss, model = train_step(data_loader=train_loader,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         rt=rt,
                                                         bad_weight=args.bad_weight)

            # testing/valid step
            test_accuracy = test_step(data_loader=test_loader,
                                      model=model)

            dev_accuracy = valid_step(data_loader=valid_loader,
                                      model=model)

            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tDev accuracy {:.4f}\tTest accuracy {:.4f}\tReduce rate {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_accuracy,
                    dev_accuracy,
                    test_accuracy,
                    rt))

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    return model


def train_eval_model(models, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True):
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long())
    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)

    ######################################################################################################
    # Train model
    model = train_model(models, train_loader, valid_loader, test_loader, args, ni)
    print('Train ended')

    ######################################################################################################
    train_results = evaluate_class(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                          'CNN', 'Train', True, plt_cm=plt_cm, plt_lables=False)
    valid_results = evaluate_class(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                          'CNN', 'Valid', True, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                         'Test', True, plt_cm=plt_cm, plt_lables=False)

    if plt_embedding and args.embedding_size <= 3:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network='CNN', saver=saver, correct=True)

    plt.close('all')
    torch.cuda.empty_cache()
    return train_results, valid_results, test_results


def main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)

            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes
    history = x_train.shape[1]

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                      norm=args.normalization)

    model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                      seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                      padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel(ae=model, classifier=classifier, name='CNN').to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
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

    for run, seed in enumerate(seeds):
        print()
        print('#' * shutil.get_terminal_size().columns)
        print('EXPERIMENT: {}/{} -- RANDOM SEED:{}'.format(run + 1, args.n_runs, seed).center(columns))
        print('#' * shutil.get_terminal_size().columns)
        print()

        args.seed = seed

        reset_seed_(seed)
        model = reset_model(model)
        # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

        test_results_main = collections.defaultdict(list)
        test_corrected_results_main = collections.defaultdict(list)
        saver_loop = SaverSlave(os.path.join(saver.path, f'seed_{seed}'))
        # saver_loop.append_str(['SEED: {}'.format(seed), '\r\n'])

        i = 0
        for ni in args.ni:
            saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
            i += 1
            # True or false
            print('+' * shutil.get_terminal_size().columns)
            print('HyperRun: %d/%d' % (i, len(args.ni)))
            print('Label noise ratio: %.3f' % ni)
            print('+' * shutil.get_terminal_size().columns)
            # saver.append_str(['#' * 100, 'Label noise ratio: %f' % ni])

            reset_seed_(seed)
            model = reset_model(model)

            Y_train, mask_train = flip_label(Y_train_clean, ni, args.label_noise)
            Y_valid, mask_valid = flip_label(Y_valid_clean, ni, args.label_noise)
            Y_test = Y_test_clean

            train_results, valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test, Y_train,
                                                                          Y_valid, Y_test, Y_train_clean,
                                                                          Y_valid_clean,
                                                                          ni, args, saver_slave,
                                                                          plt_embedding=args.plt_embedding,
                                                                          plt_cm=args.plt_cm)

            keys = list(test_results.keys())
            test_results['noise'] = ni
            test_results['seed'] = seed
            test_results['correct'] = 'SIGUA'
            test_results['losses'] = map_abg([0, 1, 0])
            # saver_loop.append_str(['Test Results:'])
            # saver_loop.append_dict(test_results)
            df_results = df_results.append(test_results, ignore_index=True)

        if args.plt_cm:
            fig_title = f"SIGUA -- Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} "
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, x='noise', hue='correct',
                         col='losses',
                         kind='bar', style='whitegrid', title=fig_title)

    if args.plt_cm:
        # Losses column should  not change here
        fig_title = f"SIGUA -- Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} "
        plot_results(df_results, keys, saver, x='noise', hue='correct', col='losses', kind='box', style='whitegrid',
                     title=fig_title)

    # boxplot_results(df_results, keys, classes, 'CNN', saver)
    # results_summary = df_results.groupby(['noise', 'correct'])[keys].describe().T
    # saver.append_str(['Results main summary', results_summary])

    remove_empty_dirs(saver.path)

    return df_results
