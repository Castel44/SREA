import subprocess
import collections
import itertools
import logging
import sys
import argparse
import os
import shutil
from time import time

import numpy as np
import pandas as pd
import torch

from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report, balanced_accuracy_score

from src.utils.metrics import evaluate_multi
from src.utils.saver import Saver

import src.utils.plotting_utils as plt

######################################################################################################################
columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################

class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_ziplen(l, n):
    if len(l) % n != 0:
        l += [l[-1]]
        return check_ziplen(l, n)
    else:
        return l


def remove_duplicates(sequence):
    unique = []
    [unique.append(item) for item in sequence if item not in unique]
    return unique


def map_abg(x):
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


def map_losstype(x):
    if x == 0:
        return 'Symm'
    else:
        return 'Asymm_{}'.format(x)


def map_abg_main(x):
    if x is None:
        return 'Variable'
    else:
        return '_'.join([str(int(j)) for j in x])


def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


def add_noise(x, sigma=0.2, mu=0.):
    noise = mu + torch.randn(x.size()) * sigma
    noisy_x = x + noise
    return noisy_x


def readable(num):
    for unit in ['', 'k', 'M']:
        if abs(num) < 1e3:
            return "%3.3f%s" % (num, unit)
        num /= 1e3
    return "%.1f%s" % (num, 'B')


# Unique labels
def categorizer(y_cont, y_discrete):
    Yd = np.diff(y_cont, axis=0)
    Yd = (Yd > 0).astype(int).squeeze()
    C = pd.Series([x + y for x, y in
                   zip(list(y_discrete[1:].astype(int).astype(str)), list((Yd).astype(str)))]).astype(
        'category')
    return C.cat.codes


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


def append_results_dict(main, sub):
    for k, v in zip(sub.keys(), sub.values()):
        main[k].append(v)
    return main


def flip_label(target, ratio, pattern=0):
    """
    Induce label noise by randomly corrupting labels
    :param target: list or array of labels
    :param ratio: float: noise ratio
    :param pattern: flag to choose which type of noise.
            0 or mod(pattern, #classes) == 0 = symmetric
            int = asymmetric
            -1 = flip
    :return:
    """
    assert 0 <= ratio < 1

    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if type(pattern) is int:
        for i in range(label.shape[0]):
            # symmetric noise
            if (pattern % n_class) == 0:
                p1 = ratio / (n_class - 1) * np.ones(n_class)
                p1[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p1)
            elif pattern > 0:
                # Asymm
                label[i] = np.random.choice([label[i], (target[i] + pattern) % n_class], p=[1 - ratio, ratio])
            else:
                # Flip noise
                label[i] = np.random.choice([label[i], 0], p=[1 - ratio, ratio])

    elif type(pattern) is str:
        raise ValueError

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask


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


def evaluate_model_multi(model, dataloder, y_true, x_true,
                         metrics=('mae', 'mse', 'rmse', 'std_ae', 'smape', 'rae', 'mbrae', 'corr', 'r2')):
    xhat, yhat = predict_multi(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    # AE
    residual = xhat - x_true
    results = evaluate_multi(actual=x_true, predicted=xhat, metrics=metrics)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted, xhat, residual, results


def evaluate_model(model, dataloder, y_true):
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
                          plt_cm=True, plt_lables=True, plt_recons=True):
    print(f'{datatype} score')
    if Y_clean is not None:
        T = confusion_matrix(Y_clean, Y)
    else:
        T = None
    results_dict = dict()

    title_str = f'{datatype} - ratio:{ni} - correct:{str(correct)}'

    results, yhat_proba, yhat, acc, f1, recons, _, ae_results = evaluate_model_multi(model, dataloader, Y, x)

    if plt_cm:
        plt.plot_cm(confusion_matrix(Y, yhat), T, network=network,
                    title_str=title_str, saver=saver)
    if plt_lables:
        plt.plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)
    if plt_recons:
        plt.plot_prediction(x, recons, nrows=5, ncols=5, figsize=(19.2, 10.80), saver=saver,
                            title=f'{datatype} data: mse:%.4f rmse:%.4f corr:%.4f R2:%.4f' % (
                                ae_results['mse'], ae_results['rmse'],
                                ae_results['corr'], ae_results['r2']), figname=f'AE_{datatype}')

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    # saver.append_str([f'{datatype}Set', 'Classification report:', results])
    # saver.append_str(['AutoEncoder results:'])
    # saver.append_dict(ae_results)
    return results_dict


def evaluate_class(model, x, Y, Y_clean, dataloader, ni, saver, network='Model', datatype='Train', correct=False,
                          plt_cm=True, plt_lables=True):
    print(f'{datatype} score')
    if Y_clean is not None:
        T = confusion_matrix(Y_clean, Y)
    else:
        T = None
    results_dict = dict()

    title_str = f'{datatype} - ratio:{ni} - correct:{str(correct)}'

    results, yhat_proba, yhat, acc, f1 = evaluate_model(model, dataloader, Y)

    if plt_cm:
        plt.plot_cm(confusion_matrix(Y, yhat), T, network=network,
                title_str=title_str, saver=saver)
    if plt_lables:
        plt.plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    #saver.append_str([f'{datatype}Set', 'Classification report:', results])
    return results_dict


def predict(model, test_data):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0]
            data = data.float().to(device)
            output = model(data)
            prediction.append(output.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    return prediction


def predict_multi(model, test_data):
    reconstruction = []
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0].float().to(device)
            out_ae, out_class, embedding = model(data)
            prediction.append(out_class.cpu().numpy())
            reconstruction.append(out_ae.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    reconstruction = np.concatenate(reconstruction, axis=0)
    return reconstruction, prediction
