import collections
import os
import shutil
from itertools import chain

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.models.AEs import MLPAE, TCNAE, LSTMAE, CNNAE
from src.models.MultiTaskClassification import MetaModel, LinClassifier, NonLinClassifier
from src.utils.saver import Saver
from src.utils.training_helper_utils import reset_seed_, evaluate_class_recons, plot_embedding, plot_results, columns, \
    mapper, remove_empty_dirs
from src.utils.utils import readable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################################################
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def reset_seed_(seed):
    # Resetting SEED to fair comparison of results
    print('Settint seed: {}'.format(seed))
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def reset_model(model):
    print('Resetting model parameters...')
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model


# Unique labels
def categorizer(y_cont, y_discrete):
    Yd = np.diff(y_cont, axis=0)
    Yd = (Yd > 0).astype(int).squeeze()
    C = pd.Series([x + y for x, y in
                   zip(list(y_discrete[1:].astype(int).astype(str)), list((Yd).astype(str)))]).astype(
        'category')
    return C.cat.codes


def RF_check(kernel_size, blocks, history):
    RF = (kernel_size - 1) * blocks * 2 ** (blocks - 1)
    print('Receptive field: {}, History window: {}'.format(RF, history))
    if RF > history:
        print('OK')
    else:
        while RF <= history:
            blocks += 1
            RF = (kernel_size - 1) * blocks * 2 ** (blocks - 1)
            print('Adding layers.. L: {}, RF:{}'.format(blocks, RF))

    print('Receptive field: {}, History window: {}, LAYERS:{}'.format(RF, history, blocks))
    return blocks


def append_results_dict(main, sub):
    for k, v in zip(sub.keys(), sub.values()):
        main[k].append(v)
    return main


def flip_label(target, ratio, pattern=0):
    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if type(pattern) is int:
        for i in range(label.shape[0]):
            if (pattern % n_class) == 0:
                p1 = ratio / (n_class - 1) * np.ones(n_class)
                p1[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p1)
            elif pattern > 0:
                # Asymm
                label[i] = np.random.choice([label[i], (target[i] + pattern) % n_class], p=[1 - ratio, ratio])
            else:
                label[i] = np.random.choice([label[i], 0], p=[1 - ratio, ratio])

    elif type(pattern) is str:
        raise ValueError

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask

def temperature(x, th_low, th_high, low_val, high_val):
    if x <= th_low:
        return low_val
    elif th_low < x < th_high:
        return (x - th_low) / (th_high - th_low) * (high_val - low_val) + low_val
    else:  # x == th_high
        return high_val


def linear_comb(w, x1, x2):
    return (1 - w) * x1 + w * x2


def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss


def train_step(data_loader, model_list: list, optimizer, criterion, rt):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()
    for x, y_hat in data_loader:
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)
        model1_loss, model2_loss = co_teaching_loss(model1_loss=model1_loss, model2_loss=model2_loss, rt=rt)

        # loss exchange
        optimizer.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer.step()

        optimizer.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer.step()

        avg_loss += (model1_loss.item() + model2_loss.item())

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


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


def train_model(models, train_loader, valid_loader, test_loader, args, tau):
    model1, model2 = models
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=args.lr, eps=1e-4)
    # learning history
    train_acc_list = []
    test_acc_list = []
    try:
        for e in range(args.epochs):
            # update reduce step
            rt = update_reduce_step(cur_step=e, num_gradual=args.num_gradual, tau=tau)

            # training step
            train_accuracy, avg_loss, model_list = train_step(data_loader=train_loader,
                                                              model_list=[model1, model2],
                                                              optimizer=optimizer,
                                                              criterion=criterion,
                                                              rt=rt)
            model1, model2 = model_list

            # testing/valid step
            test_accuracy = test_step(data_loader=test_loader,
                                      model=model1)

            dev_accuracy = valid_step(data_loader=valid_loader,
                                      model=model1)

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

    return model1


def boxplot_results(data, keys, classes, network, saver):
    n = len(keys)
    x = 'noise'
    hue = 'correct'
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 7 + (n * 0.1)), sharex='all')
    for ax, k in zip(axes, keys):
        sns.boxplot(x=x, y=k, hue=hue, data=data, ax=ax)
        ax.grid(True, alpha=0.2)
    axes[0].set_title('Model:{} classes:{} - Co-Teaching'.format(network, classes))
    fig.tight_layout()
    saver.save_fig(fig, 'boxplot')


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
    train_results = evaluate_class_recons(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                          args.network, 'Train', True, plt_cm=plt_cm, plt_lables=False)
    valid_results = evaluate_class_recons(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                          args.network, 'Valid', True, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class_recons(model, x_test, Y_test, None, test_loader, ni, saver, args.network,
                                         'Test', True, plt_cm=plt_cm, plt_lables=False)

    if plt_embedding and args.embedding_size <= 3:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network=args.network, saver=saver, correct=True)

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
    if args.nonlin_classifier:
        classifier1 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                       norm=args.normalization)
        classifier2 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                       norm=args.normalization)
    else:
        classifier1 = LinClassifier(args.embedding_size, classes)
        classifier2 = LinClassifier(args.embedding_size, classes)

    if args.network == 'MLP':
        # Reshape data for MLP
        x_train = np.hstack([(x_train[:, :, i]) for i in range(x_train.shape[2])])
        x_valid = np.hstack([(x_valid[:, :, i]) for i in range(x_valid.shape[2])])
        x_test = np.hstack([(x_test[:, :, i]) for i in range(x_test.shape[2])])

        model1 = MLPAE(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, hidden_neurons=args.neurons,
                       hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                       normalization=args.normalization).to(device)
        model2 = MLPAE(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, hidden_neurons=args.neurons,
                       hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                       normalization=args.normalization).to(device)

    elif args.network == 'TCN':
        stacked_layers = RF_check(args.kernel_size, args.stack, history)

        model1 = TCNAE(input_size=x_train.shape[2], num_filters=args.filter_number, embedding_dim=args.embedding_size,
                       seq_len=x_train.shape[1], num_stack=stacked_layers, kernel_size=args.kernel_size,
                       dropout=args.dropout, normalization=args.normalization).to(device)
        model2 = TCNAE(input_size=x_train.shape[2], num_filters=args.filter_number, embedding_dim=args.embedding_size,
                       seq_len=x_train.shape[1], num_stack=stacked_layers, kernel_size=args.kernel_size,
                       dropout=args.dropout, normalization=args.normalization).to(device)

    elif args.network == 'CNN':
        model1 = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                       seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                       padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
        model2 = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                       seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                       padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    elif args.network == 'LSTM':
        model1 = LSTMAE(seq_len_out=x_train.shape[1], n_features=x_train.shape[2],
                        n_layers=args.rnn_layers, embedding_dim=args.embedding_size).to(device)
        model2 = LSTMAE(seq_len_out=x_train.shape[1], n_features=x_train.shape[2],
                        n_layers=args.rnn_layers, embedding_dim=args.embedding_size).to(device)
    else:
        raise NotImplementedError

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model1 = MetaModel(ae=model1, classifier=classifier1, n_out=args.n_out, name=args.network).to(device)
    model2 = MetaModel(ae=model2, classifier=classifier2, n_out=args.n_out, name=args.network).to(device)
    models = [model1, model2]

    nParams = sum([p.nelement() for p in model1.parameters()])
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

    for run, seed in enumerate(seeds):
        print()
        print('#' * shutil.get_terminal_size().columns)
        print('EXPERIMENT: {}/{} -- RANDOM SEED:{}'.format(run + 1, args.n_runs, seed).center(columns))
        print('#' * shutil.get_terminal_size().columns)
        print()

        args.seed = seed

        reset_seed_(seed)
        models = [reset_model(m) for m in models]
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
            models = [reset_model(m) for m in models]

            Y_train, mask_train = flip_label(Y_train_clean, ni, args.label_noise)
            Y_valid, mask_valid = flip_label(Y_valid_clean, ni, args.label_noise)
            Y_test = Y_test_clean

            train_results, valid_results, test_results = train_eval_model(models, x_train, x_valid, x_test, Y_train,
                                                                          Y_valid, Y_test, Y_train_clean,
                                                                          Y_valid_clean,
                                                                          ni, args, saver_slave,
                                                                          plt_embedding=args.plt_embedding,
                                                                          plt_cm=args.plt_cm)

            keys = list(test_results.keys())
            test_results['noise'] = ni
            test_results['seed'] = seed
            test_results['correct'] = 'Co-teaching'
            test_results['losses'] = mapper(args.abg)
            # saver_loop.append_str(['Test Results:'])
            # saver_loop.append_dict(test_results)
            df_results = df_results.append(test_results, ignore_index=True)

        if args.plt_cm:
            fig_title = f"CO-TEACHING -- Dataset: {args.dataset} - Model: {args.network} - classes:{classes} - runs:{args.n_runs} "
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, x='noise', hue='correct',
                         col='losses',
                         kind='bar', style='whitegrid', title=fig_title)
    if args.plt_cm:
        # Losses column should  not change here
        fig_title = f"CO-TEACHING -- Dataset: {args.dataset} - Model: {args.network} - classes:{classes} - runs:{args.n_runs} "
        plot_results(df_results, keys, saver, x='noise', hue='correct', col='losses', kind='box', style='whitegrid',
                     title=fig_title)

    # boxplot_results(df_results, keys, classes, args.network)
    # results_summary = df_results.groupby(['noise', 'correct'])[keys].describe().T
    # saver.append_str(['Results main summary', results_summary])

    remove_empty_dirs(saver.path)

    return df_results
