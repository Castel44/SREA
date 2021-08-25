import collections
import itertools
import os
import shutil
from time import time
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn import cluster
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.model import MLPAE, TCNAE, LSTMAE, CNNAE
from src.models.MultiTaskClassification import MetaModel, LinClassifier, NonLinClassifier
from src.utils.metrics import evaluate_multi
from src.utils.robust_losses import TaylorCrossEntropy, Unhinged, PHuberCrossEntropy, PHuberGeneralizedCrossEntropy, \
    GeneralizedCrossEntropy
from src.utils.saver import Saver
from src.utils.torch_utils import plot_loss
from src.utils.torch_utils import predict, predict_multi
from src.utils.utils import readable
from src.utils.utils_postprocess import plot_prediction

from src.utils.utils_mixup_BMM import *

columns = shutil.get_terminal_size().columns
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


def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


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


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze()
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        reg = self.regularization(reduction='sum')
        return loss + self.rho * reg

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
                max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def create_hard_labels(embedding, centers, y_obs, yhat_hist, w_yhat, w_c, w_obs, classes):
    # TODO: add label temporal dynamics

    # yhat from previous metwork prediction. - Network Ensemble
    # TODO: weighted average. Exponential decay??
    decay = np.arange(yhat_hist.size(-1))
    decay = -decay / yhat_hist.size(-1) + 1
    decay = torch.Tensor(decay).to(device)
    yhat_hist = yhat_hist * decay
    yhat = yhat_hist.mean(dim=-1) * w_yhat

    # Label from clustering
    distance_centers = torch.cdist(embedding, centers)
    yc = F.softmin(distance_centers, dim=1).detach() * w_c

    # Observed - given - label (noisy)
    yobs = F.one_hot(y_obs, num_classes=classes).float() * w_obs

    # Label combining
    ystar = (yhat + yc + yobs) / 3
    ystar = torch.argmax(ystar, dim=1)
    return ystar


def plot_pred_labels(y_true, y_hat, accuracy, residuals=None, dataset='Train', saver=None):
    # TODO: add more metrics
    # Plot data as timeseries
    gridspec_kw = {'width_ratios': [1], 'height_ratios': [3, 1]}

    if residuals is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex='all', gridspec_kw=gridspec_kw)

        ax2.plot(residuals ** 2, marker='o', color='red', label='Squared Residual Error', alpha=0.5, markersize='2')
        # ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend(loc=1)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    ax1.plot(y_true.ravel(), linestyle='-', marker='o', color='black', label='True', markersize='2')
    ax1.plot(y_hat.ravel(), linestyle='--', marker='o', color='red', label='Prediction', alpha=0.5,
             markersize='2')
    ax1.set_title('%s data: top1 acc: %.4f' % (dataset, accuracy))
    ax1.legend(loc=1)

    fig.tight_layout()
    saver.save_fig(fig, name='%s series' % dataset)


def train_model(model, train_loader, valid_loader, test_loader, mixup, bootbeta, args, saver=None):
    network = model.get_name()
    milestones = args.M
    num_classes = args.nbins

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} MixUp - {} BootBeta'.format(network, mixup, bootbeta).center(
        columns)
    print(s)
    print('-' * shutil.get_terminal_size().columns)
    #saver.append_str(['*' * 100, s, '*' * 100])

    optimizer = eval(args.optimizer)(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=args.lr, weight_decay=args.l2penalty, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    bmm_model = bmm_model_maxLoss = bmm_model_minLoss = cont = k = 0

    bootstrap_ep_std = milestones[0] + 5 + 1  # the +1 is because the conditions are defined as ">" or "<" not ">="
    guidedMixup_ep = 60

    if args.Mixup == 'Dynamic':
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        bootstrap_ep_mixup = milestones[0] + 5 + 1

    countTemp = 1

    temp_length = 100 - bootstrap_ep_mixup
    try:
        for epoch in range(1, args.epochs + 1):
            # train

            ### Standard CE training (without mixup) ###
            if mixup == "None":
                print('\t##### Doing standard training with cross-entropy loss #####')
                loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer,
                                                                           epoch)

            ### Mixup ###
            if mixup == "Static":
                alpha = args.alpha
                if epoch < bootstrap_ep_mixup:
                    print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer,
                                                                        epoch,
                                                                        32)

                else:
                    if bootbeta == "Hard":
                        print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(
                            bootstrap_ep_mixup))
                        loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device,
                                                                                         train_loader,
                                                                                         optimizer, epoch, \
                                                                                         alpha, bmm_model,
                                                                                         bmm_model_maxLoss,
                                                                                         bmm_model_minLoss,
                                                                                         args.reg_term,
                                                                                         num_classes)
                    elif bootbeta == "Soft":
                        print("\t##### Doing SOFT BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(
                            bootstrap_ep_mixup))
                        loss_per_epoch, acc_train_per_epoch_i = train_mixUp_SoftBootBeta(args, model, device,
                                                                                         train_loader,
                                                                                         optimizer, epoch, \
                                                                                         alpha, bmm_model,
                                                                                         bmm_model_maxLoss,
                                                                                         bmm_model_minLoss,
                                                                                         args.reg_term,
                                                                                         num_classes)

                    else:
                        print('\t##### Doing NORMAL mixup #####')
                        loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader,
                                                                            optimizer,
                                                                            epoch,
                                                                            32)

            ## Dynamic Mixup ##
            if mixup == "Dynamic":
                alpha = args.alpha
                if epoch < guidedMixup_ep:
                    print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(guidedMixup_ep - 1))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer,
                                                                        epoch,
                                                                        32)

                elif epoch < bootstrap_ep_mixup:
                    print('\t##### Doing Dynamic mixup from epoch {0} #####'.format(guidedMixup_ep))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_Beta(args, model, device, train_loader,
                                                                             optimizer,
                                                                             epoch, alpha, bmm_model, \
                                                                             bmm_model_maxLoss, bmm_model_minLoss)
                else:
                    print(
                        "\t##### Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from the epoch {0} #####".format(
                            bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i, countTemp, k = train_mixUp_SoftHardBetaDouble(args, model,
                                                                                                         device,
                                                                                                         train_loader,
                                                                                                         optimizer, \
                                                                                                         epoch,
                                                                                                         bmm_model,
                                                                                                         bmm_model_maxLoss,
                                                                                                         bmm_model_minLoss, \
                                                                                                         countTemp, k,
                                                                                                         temp_length,
                                                                                                         args.reg_term,
                                                                                                         num_classes)
            scheduler.step()

            ### Training tracking loss
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, model, device, valid_loader, epoch, bmm_model, bmm_model_maxLoss,
                                    bmm_model_minLoss)

            # test
            loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)


    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    return model


def eval_model(model, loader, list_loss, coeffs):
    loss_ae, loss_class, loss_centroids = list_loss
    alpha, beta, gamma = coeffs
    losses = []
    accs = []

    with torch.no_grad():
        model.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            target = Variable(target.long()).to(device)
            batch_size = inputs.size(0)

            out_AE, out_class = model(inputs)
            embedding = model.encoder(inputs).squeeze()
            ypred = torch.max(F.softmax(out_class, dim=1), dim=1)[1]

            loss_recons_ = loss_ae(out_AE, inputs)
            loss_class_ = loss_class(out_class, target)
            loss_cntrs_ = loss_centroids(embedding, target)
            loss = alpha * loss_recons_ + beta * loss_class_.mean() + gamma * loss_cntrs_.mean()

            losses.append(loss.data.item())

            accs.append((ypred == target).sum().item() / batch_size)

    return np.array(losses).mean(), 100 * np.average(accs)


def plot_label_insight(data, target, saver=None):
    try:
        data = data.squeeze(-1)
    except:
        try:
            data = np.hstack([(data[:, :, i]) for i in range(data.shape[2])])
        except:
            pass

    n_classes = len(np.unique(target))

    fig, axes = plt.subplots(nrows=n_classes, ncols=1, figsize=(19.20, 10.80))

    # Plot class centroid / examples
    D = {}
    for i in np.unique(target):
        D[i] = {'mu': np.mean(data[target == i], axis=0).ravel(),
                'std': np.std(data[target == i], axis=0).ravel(),
                # 'median': np.median(train_data[target_discrete == i], axis=0).ravel(),
                }

    for i in range(n_classes):
        axes[i].plot(D[i]['mu'], '-o', label='mean', color='tab:blue')
        # axes[i][1].plot(D[i]['median'], '-o', label='median')
        axes[i].fill_between(range(D[i]['mu'].shape[0]), D[i]['mu'] - D[i]['std'], D[i]['mu'] + D[i]['std'],
                             alpha=0.33, label='stddev', color='tab:green')
        axes[i].legend(loc=1)
        axes[i].grid()
        axes[i].set_title('Class {}'.format(i))
    fig.tight_layout()

    if saver:
        saver.save_fig(fig, 'Label_Insight')


def plot_label_insight_v2(data, target_continous, train_data, target_discrete, history=36, future=6,
                          saver=None):
    # TODO: Remove those ugly try: exept:
    try:
        train_data = train_data.squeeze(-1)
    except:
        try:
            train_data = np.hstack([(train_data[:, :, i]) for i in range(train_data.shape[2])])
        except:
            pass

    n_classes = len(np.unique(target_discrete))
    data_min = np.min(data)
    data_max = np.max(data)

    fig, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(19.20, 10.80))
    # Plot input Data
    gs = axes[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axes[:2, 0]:
        ax.remove()
    axdata = fig.add_subplot(gs[:2, 0])
    axdata.plot(data[:10 * history])
    axdata.fill_between(range(history), data_min, data_max, alpha=0.25, color='green', label='Input Window')
    axdata.fill_between(range(history - 1, history + future), data_min, data_max, alpha=0.25, color='red',
                        label='Target Window')
    axdata.legend(loc=1)
    # axdata.grid()
    axdata.set_title('Input Data - Full Raw')

    # Plot Labels
    cmap = plt.cm.jet
    points = 3 * history
    gs = axes[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axes[2:, 0]:
        ax.remove()
    axtarget = fig.add_subplot(gs[2:, 0])
    line = axtarget.scatter(np.arange(points), target_continous[:points], c=target_discrete[:points], cmap=cmap)
    axtarget.grid()
    plt.colorbar(line, values=np.unique(target_discrete))
    axtarget.set_title('Target Label')

    # Plot class centroid / examples
    D = {}
    for i in np.unique(target_discrete):
        D[i] = {'mu': np.mean(train_data[target_discrete == i], axis=0).ravel(),
                'std': np.std(train_data[target_discrete == i], axis=0).ravel(),
                # 'median': np.median(train_data[target_discrete == i], axis=0).ravel(),
                }

    for i in range(n_classes):
        axes[i][1].plot(D[i]['mu'], '-o', label='mean', color=cmap(i / (n_classes - 1)))
        # axes[i][1].plot(D[i]['median'], '-o', label='median')
        axes[i][1].fill_between(range(D[i]['mu'].shape[0]), D[i]['mu'] - D[i]['std'], D[i]['mu'] + D[i]['std'],
                                alpha=0.33, label='stddev', color=cmap(i / (n_classes - 1)))
        axes[i][1].legend(loc=1)
        axes[i][1].grid()
        axes[i][1].set_title('Class {}'.format(i))
    fig.tight_layout()

    if saver:
        saver.save_fig(fig, 'Label_Insight')


def plot_test_reuslts(test: dict, test_correct: dict, ni_list: list, classes: int, network: str, mixup: str,
                      bootbeta: str,
                      seed: int, saver: object) -> None:
    if test.keys() != test_correct.keys():
        print('Plain and Corrected dict_keys are different. Plotting only test...')
        test_correct = {k: [] for k in test.keys()}

    n = len(test.keys())

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, 5 + (n * 0.1)), sharex='all')
    for ax, (key, tst), tst_corr in zip(axes, test.items(), test_correct.values()):
        ax.plot(tst, '--o', label='Test (Naive)')
        ax.plot(tst_corr, '--s', label='Test (Corrected)')
        ax.set_ylim([0, 1])

        ax.set_ylabel('{}'.format(key))
        ax.set_xticks([i for i in range(len(ni_list))])
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[-1].set_xticklabels(ni_list)
    axes[-1].set_xlabel('Label Noise ratio')
    axes[0].set_title(
        'Model:{} n_classes:{} - Mixup:{} - BootBeta:{} - rng_seed:{}'.format(network, classes, mixup, bootbeta, seed))
    fig.tight_layout()

    saver.save_fig(fig, 'run_results')


def boxplot_results(data, keys, classes, network, mixup, bootbeta, saver):
    n = len(keys)
    x = 'noise'
    hue = 'correct'
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 7 + (n * 0.1)), sharex='all')
    for ax, k in zip(axes, keys):
        sns.boxplot(x=x, y=k, hue=hue, data=data, ax=ax)
        ax.grid(True, alpha=0.2)
    axes[0].set_title('Model:{} classes:{} - Mixup:{} - BootBeta:{}'.format(network, classes, mixup, bootbeta))
    fig.tight_layout()
    saver.save_fig(fig, 'boxplot')


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


def evaluate_model_multi(model, dataloder, y_true, x_true,
                         metrics=('mae', 'mse', 'rmse', 'std_ae', 'smape', 'rae', 'mbrae', 'corr', 'r2')):
    yhat = predict(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted


def evaluate_class_recons(model, x, Y, Y_clean, dataloader, ni, saver, network='Model', datatype='Train', correct=False,
                          plt_cm=True, plt_lables=True):
    print(f'{datatype} score')
    if Y_clean is not None:
        T = confusion_matrix(Y_clean, Y)
    else:
        T = None
    results_dict = dict()

    title_str = f'{datatype} - ratio:{ni} - correct:{str(correct)}'

    results, yhat_proba, yhat, acc, f1 = evaluate_model_multi(model, dataloader, Y, x)

    if plt_cm:
        plot_cm(confusion_matrix(Y, yhat), T, network=network,
                title_str=title_str, saver=saver)
    if plt_lables:
        plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    #saver.append_str([f'{datatype}Set', 'Classification report:', results])
    return results_dict


def plot_cm(cm, T=None, network='Net', title_str='', saver=None):
    classes = cm.shape[0]
    acc = np.diag(cm).sum() / cm.sum()
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if T is not None:
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4 + classes / 2.5, 4 + classes / 1.25))
        T_norm = T.astype('float') / T.sum(axis=1)[:, np.newaxis]
        # Transition matrix ax
        sns.heatmap(T_norm, annot=T_norm, cmap=plt.cm.YlGnBu, cbar=False, ax=ax2, linecolor='black', linewidths=0)
        ax2.set(ylabel='Noise Transition Matrix')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4 + classes / 2.5, 4 + classes / 2.5))

    # Cm Ax
    sns.heatmap(cm_norm, annot=None, cmap=plt.cm.YlGnBu, cbar=False, ax=ax, linecolor='black', linewidths=0)
    # ax.imshow(cm_norm, aspect='auto', interpolation='nearest', cmap=plt.cm.YlGnBu)
    # ax.matshow(cm_norm, cmap=plt.cm.Blues)

    ax.set(title=f'Model:{network} - Accuracy:{100 * acc:.1f}% - {title_str}',
           ylabel='Confusion Matrix (Predicted / True)',
           xlabel=None)
    # ax.set_ylim([1.5, -0.5])
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, '%d (%.2f)' % (cm[i, j], cm_norm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()

    if saver:
        saver.save_fig(fig, f'CM_{title_str}')


def plot_embedding(model, train_loader, valid_loader, Y_train_clean, Y_valid_clean, Y_train, Y_valid,
                   saver, network='Model', correct=False):
    print('Plot Embedding...')
    # Embeddings
    train_embedding = predict(model, train_loader).squeeze()
    valid_embedding = predict(model, valid_loader).squeeze()
    classes = len(np.unique(Y_train_clean))

    ttl = f'{network} - Embedding'
    n_comp = 2
    if train_embedding.shape[-1] > 3:
        from umap import UMAP
        trs = UMAP(n_components=n_comp, n_neighbors=50, min_dist=0.01, metric='euclidean')
        ttl = 'UMAP'
        train_embedding2d = trs.fit_transform(train_embedding)
        valid_embedding2d = trs.transform(valid_embedding)
    else:
        train_embedding2d = train_embedding
        valid_embedding2d = valid_embedding

    cmap = 'jet'
    COL = MplColorHelper(cmap, 0, classes)

    plt.figure(figsize=(8, 6))
    if train_embedding2d.shape[1] == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    l0 = ax.scatter(*train_embedding2d.T, s=50, alpha=0.5, marker='.', label='Train',
                    c=COL.get_rgb(Y_train_clean),
                    edgecolors=COL.get_rgb(Y_train))
    l1 = ax.scatter(*valid_embedding2d.T, s=50, alpha=0.5, marker='^', label='Valid',
                    c=COL.get_rgb(Y_valid_clean),
                    edgecolors=COL.get_rgb(Y_valid))
    lines = [l0, l1] + [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=10) for c in
                        [COL.get_rgb(i) for i in np.unique(Y_train_clean.astype(int))]]
    labels = [l0.get_label(), l1.get_label()] + [i for i in range(len(lines))]
    ax.legend(lines, labels)
    ax.set_title(ttl)
    plt.tight_layout()
    saver.save_fig(plt.gcf(), name=f'{network}_latent_{str(correct)}')


def plot_hists_ephocs(loss, mask, auc=False, nrows=3, ncols=3, net='MLP', classes=5, saver=None, ni=None, pred_ni=None):
    '''
    mask : mislabel mask. 1: wrong label, 0: correct label
    '''
    if auc:
        data = loss.cumsum(axis=0)
        data_type = 'AUC'
    else:
        data = loss
        data_type = 'LOSS'

    plots = int(nrows * ncols)
    epochs = data.shape[0]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(19.20, 10.80))
    for i, ax in enumerate(axes.flatten()):
        id = int((epochs - 1) * i * (1 / (plots - 1)))
        sns.distplot(data[id], kde=True, hist=False, rug=False, ax=ax,
                     label='Joint', kde_kws={"color": "black", "linestyle": "--", "lw": 4})
        sns.distplot(data[id][~mask.astype(bool)], kde=False, hist=True, rug=False, norm_hist=True, ax=ax,
                     label='Clean', kde_kws={'alpha': 0.6, "lw": 3, 'color': 'tab:blue'},
                     hist_kws={'alpha': 0.3, 'color': 'tab:blue'})
        sns.distplot(data[id][mask.astype(bool)], kde=False, hist=True, rug=False, norm_hist=True, ax=ax,
                     label='Mislabled', kde_kws={'alpha': 0.6, "lw": 3, 'color': 'tab:orange'},
                     hist_kws={'alpha': 0.3, 'color': 'tab:orange'})
        ax.legend()
        ax.set(title=f'Epoch {id + 1}/{epochs} ({i * (100 / (plots - 1)):.1f}%)')
    fig.suptitle(
        'TRAINING {} - Net:{} - Classes:{} - True error_rate:{}. Predicted:{:.3f}'.format(data_type, net, classes, ni,
                                                                                          pred_ni))
    fig.tight_layout()
    if saver:
        saver.save_fig(fig, '{}_dist'.format(data_type))


def visualize_training_loss(train_losses, train_idxs, mask_train, network, classes, ni, saver, correct=False):
    print('Visualize training losses..')
    train_losses = np.array([train_losses[i][train_idxs.argsort()[i]] for i in range(len(train_idxs))])

    plot_hists_ephocs(train_losses, mask_train, auc=False, nrows=3, ncols=3, net=network, classes=classes,
                      saver=saver, ni=ni, pred_ni=0)

    ### Sample Loss
    fig, ax = plt.subplots()
    clean_med = np.median(train_losses[:, ~mask_train.astype(bool)], axis=1)
    clean_q75, clean_q25 = np.percentile(train_losses[:, ~mask_train.astype(bool)], [75, 25], axis=1)
    mislabled_med = np.median(train_losses[:, mask_train.astype(bool)], axis=1)
    mislabled_q75, mislabled_q25 = np.percentile(train_losses[:, mask_train.astype(bool)], [75, 25], axis=1)

    ax.plot(clean_med, label='Clean', color='tab:blue')
    ax.fill_between(range(clean_med.shape[0]), clean_q25, clean_q75, alpha=0.25,
                    color='tab:blue')
    ax.plot(mislabled_med, label='Mislabled', color='tab:orange')
    ax.fill_between(range(mislabled_med.shape[0]), mislabled_q75, mislabled_q25,
                    alpha=0.25, color='tab:orange')
    ax.set(title=r'Train Loss function (median $\pm$ IRQ25:75) - noise_ratio:{}'.format(ni),
           xlabel='Epochs',
           ylabel=r'$\mathcal{L}_c(x, y)$')
    ax.grid()
    ax.legend()
    saver.save_fig(fig, f'Loss_{ni}_{str(correct)}')


def train_eval_model(model, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     ni, args, mixup, bootbeta, saver, correct_labels=True, plt_embedding=True, plt_cm=True):
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
    model = train_model(model, train_loader, valid_loader, test_loader, mixup, bootbeta, args, saver)
    print('Train ended')

    ######################################################################################################
    train_results = evaluate_class_recons(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                          args.network, 'Train', correct_labels, plt_cm=plt_cm, plt_lables=False)
    valid_results = evaluate_class_recons(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                          args.network, 'Valid', correct_labels, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class_recons(model, x_test, Y_test, None, test_loader, ni, saver, args.network,
                                         'Test', correct_labels, plt_cm=plt_cm, plt_lables=False)

    if plt_embedding and args.embedding_size <= 3:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network=args.network, saver=saver, correct=correct_labels)

    plt.close('all')
    torch.cuda.empty_cache()
    return train_results, valid_results, test_results


def mapper(x):
    if x == [0, 1, 0]:
        return r'$\mathcal{L}_c$'
    elif x == [1, 0, 0]:
        return r'$\mathcal{L}_{ae}$'
    elif x == [1, 1, 0]:
        return r'$\mathcal{L}_c + \mathcal{L}_{ae}$'
    elif x == [0, 1, 1]:
        return r'$\mathcal{L}_c + \mathcal{L}_{cc}$'
    elif x == [1, 1, 1]:
        return r'$\mathcal{L}_c + \mathcal{L}_{ae} + \mathcal{L}_{cc}$'
    else:
        raise ValueError


def main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)

            self.path = path
            self.makedir_()
            #self.make_log()

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

    elif args.network == 'CNN':
        model_ae = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                         seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                         padding=args.padding, dropout=args.dropout, normalization=args.normalization)

    elif args.network == 'LSTM':
        # TODO: fix
        model_ae = LSTMAE(seq_len_out=x_train.shape[1], n_features=x_train.shape[2],
                          n_layers=args.rnn_layers, embedding_dim=args.embedding_size).to(device)
    else:
        raise NotImplementedError

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel(ae=model_ae, classifier=classifier, n_out=args.n_out, name=args.network).to(device)
    # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

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
        #saver_loop.append_str(['SEED: {}'.format(seed), '\r\n'])

        i = 0
        for ni in args.ni:
            saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
            for correct_labels in args.correct:
                i += 1
                # True or false
                print('+' * shutil.get_terminal_size().columns)
                print('HyperRun: %d/%d' % (i, len(args.ni) * len(args.correct)))
                print('Label noise ratio: %.3f' % ni)
                print('Correct labels:', correct_labels)
                print('+' * shutil.get_terminal_size().columns)
                #saver.append_str(['#' * 100, 'Label noise ratio: %f' % ni, 'Correct Labels: %s' % correct_labels])

                reset_seed_(seed)
                model = reset_model(model)

                Y_train, mask_train = flip_label(Y_train_clean, ni * .01, args.label_noise)
                Y_valid, mask_valid = flip_label(Y_valid_clean, ni * .01, args.label_noise)
                Y_test = Y_test_clean

                if correct_labels.lower() == 'none':
                    mixup, bootbeta = 'None', 'None'
                elif correct_labels == 'Mixup':
                    mixup = 'Static'
                    bootbeta = 'None'
                else:
                    mixup = args.Mixup
                    bootbeta = args.BootBeta

                # Re-load initial weights
                # model.load_state_dict(torch.load(os.path.join(saver.path, 'initial_weight.pt')))

                train_results, valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test, Y_train,
                                                                              Y_valid, Y_test, Y_train_clean,
                                                                              Y_valid_clean,
                                                                              ni, args, mixup, bootbeta, saver_slave,
                                                                              correct_labels=correct_labels,
                                                                              plt_embedding=args.plt_embedding,
                                                                              plt_cm=args.plt_cm)

                keys = list(test_results.keys())
                test_results['noise'] = ni*.01
                test_results['seed'] = seed
                test_results['correct'] = str(correct_labels)
                test_results['losses'] = mapper(args.abg)
                #saver_loop.append_str(['Test Results:'])
                #saver_loop.append_dict(test_results)
                df_results = df_results.append(test_results, ignore_index=True)

        if args.plt_cm:
            fig_title = f"Dataset: {args.dataset} - Model: {args.network} - classes:{classes} - runs:{args.n_runs} " \
                        f"- MixUp:{args.Mixup} - BootBeta:{args.BootBeta}"
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, x='noise', hue='correct', col='losses',
                         kind='bar', style='whitegrid',  title=fig_title)

    if args.plt_cm:
        # Losses column should  not change here
        fig_title = f"Dataset: {args.dataset} - Model: {args.network} - classes:{classes} - runs:{args.n_runs} " \
                    f"- MixUp:{args.Mixup} - BootBeta:{args.BootBeta}"
        plot_results(df_results, keys, saver, x='noise', hue='correct', col='losses', kind='box', style='whitegrid',
                     title=fig_title)

    #boxplot_results(df_results, keys, classes, args.network, args.Mixup, args.BootBeta, saver)

    #results_summary = df_results.groupby(['noise', 'correct'])[keys].describe().T
    #saver.append_str(['Results main summary', results_summary])

    remove_empty_dirs(saver.path)

    return df_results
