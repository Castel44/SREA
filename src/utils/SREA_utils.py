import collections
import itertools
import logging
import sys
import os
import shutil
from time import time

from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn import cluster
from sklearn.model_selection import train_test_split

from src.models.model import CNNAE
from src.models.MultiTaskClassification import AEandClass, LinClassifier, NonLinClassifier
from src.utils.saver import Saver
from src.utils.utils import readable
from src.utils.log_utils import StreamToLogger
from src.utils.ucr_datasets import load_data as load_ucr
from src.utils.utils import cluster_accuracy, evaluate_class_recons, reset_seed_, reset_model, SaverSlave, flip_label, \
    append_results_dict, map_losstype, map_abg, remove_empty_dirs
from src.utils.plotting_utils import plot_loss, plot_embedding, visualize_training_loss, plot_results

columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_comb(w, x1, x2):
    return (1 - w) * x1 + w * x2


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    """
    Centroid loss - Constraint Clustering loss of SREA
    """

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


def temperature(x, th_low, th_high, low_val, high_val):
    if x < th_low:
        return low_val
    elif th_low <= x < th_high:
        return (x - th_low) / (th_high - th_low) * (high_val - low_val) + low_val
    else:  # x == th_high
        return high_val


def create_hard_labels(embedding, centers, y_obs, yhat_hist, w_yhat, w_c, w_obs, classes):
    # TODO: add label temporal dynamics

    # yhat from previous metwork prediction. - Network Ensemble
    steps = yhat_hist.size(-1)
    decay = torch.arange(0, steps, 1).float().to(device)
    decay = torch.exp(-decay / 2)
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


def train_model(model, train_data, valid_data, epochs, correct, args, saver=None, plot_loss_flag=True):
    # Init variables
    network = model.get_name()
    milestone = args.M
    alpha, beta, gamma = args.abg
    rho = args.class_reg
    epsilon = args.entropy_reg
    history_track = args.track
    correct_start = args.correct_start
    correct_end = args.correct_end
    init_centers = args.init_centers
    classes = args.nbins

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    # Init losses
    loss_class = nn.CrossEntropyLoss(reduction='none')
    loss_ae = nn.MSELoss(reduction='mean')
    loss_centroids = CentroidLoss(args.embedding_size, classes, reduction='none').to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
        lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.5)

    p = torch.ones(classes).to(device) / classes
    kmeans = cluster.KMeans(n_clusters=classes, random_state=args.seed)
    yhat_hist = torch.zeros(train_data.dataset.tensors[0].size(0), classes, history_track).to(device)

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} LOSS - Correction: {}'.format(network, loss_class._get_name(), str(correct))
    print(s)
    print('-' * shutil.get_terminal_size().columns)

    # Train loop
    # Force exit with Ctrl + C (Keyboard interrupt command)
    try:
        all_losses = []
        all_indices = []

        for idx_epoch in range(1, epochs + 1):
            epochstart = time()
            train_loss = []
            train_acc = []
            train_acc_corrected = []
            epoch_losses = torch.Tensor()
            epoch_indices = torch.Tensor()

            # KMeans after the first milestone - Training WarmUp
            if idx_epoch == init_centers:
                # Init cluster centers with KMeans
                embedding = []
                targets = []
                with torch.no_grad():
                    model.eval()
                    loss_centroids.eval()
                    for data, target, _ in train_data:
                        data = data.to(device)
                        output = model.encoder(data)
                        embedding.append(output.squeeze().cpu().numpy())
                        targets.append(target.numpy())
                embedding = np.concatenate(embedding, axis=0)
                targets = np.concatenate(targets, axis=0)
                predicted = kmeans.fit_predict(embedding)
                reassignment, accuracy = cluster_accuracy(targets, predicted)
                # predicted_ordered = np.array(list(map(lambda x: reassignment[x], predicted)))
                # Center reordering. Swap keys and values and sort by keys.
                cluster_centers = kmeans.cluster_centers_[
                    list(dict(sorted({y: x for x, y in reassignment.items()}.items())).values())]
                cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True).to(device)
                with torch.no_grad():
                    # initialise the cluster centers
                    loss_centroids.state_dict()["centers"].copy_(cluster_centers)

            # Train
            model.train()
            loss_centroids.train()
            for data, target, data_idx in train_data:
                target = target.to(device)
                data = data.to(device)
                batch_size = data.size(0)

                # Forward
                optimizer.zero_grad()
                out_AE, out_class, embedding = model(data)
                embedding = embedding.squeeze()

                # Accuracy on noisy labels
                prob = F.softmax(out_class, dim=1)
                prob_avg = torch.mean(prob, dim=0)
                train_acc.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)
                loss_noisy_labels = loss_class(out_class, target).detach()

                # Track predictions
                alpha_, beta_, gamma_, epsilon_, rho_ = alpha, beta, gamma, epsilon, rho
                w_yhat, w_c, w_obs = 0, 0, 0

                # Correct labels
                if correct:
                    w_yhat = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=0,
                                         high_val=1 * beta)  # Pred
                    w_c = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=0,
                                      high_val=1 * gamma)  # Centers
                    w_obs = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=1,
                                        high_val=0)  # Observed

                    beta_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                        low_val=0, high_val=beta)  # Class
                    gamma_ = temperature(idx_epoch, th_low=init_centers, th_high=correct_start, low_val=0,
                                         high_val=gamma)  # Centers
                    rho_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                       low_val=0, high_val=rho * beta_)  # Lp
                    epsilon_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                           low_val=0, high_val=epsilon * beta_)  # Le

                    ystar = create_hard_labels(embedding, loss_centroids.centers, target, yhat_hist[data_idx],
                                               w_yhat, w_c, w_obs, classes)
                    target = ystar
                else:
                    gamma_ = temperature(idx_epoch, th_low=init_centers, th_high=init_centers, low_val=0,  # Centers
                                         high_val=gamma)
                    rho_ *= beta
                    epsilon_ *= beta

                loss_cntrs_ = loss_centroids(embedding, target)
                loss_class_ = loss_class(out_class, target)
                loss_recons_ = loss_ae(out_AE, data)

                L_p = -torch.sum(torch.log(prob_avg) * p)  # Distribution regularization
                L_e = -torch.mean(torch.sum(prob * F.log_softmax(out_class, dim=1), dim=1))  # Entropy regularization

                loss = alpha_ * loss_recons_ + beta_ * loss_class_.mean() + gamma_ * loss_cntrs_.mean() + \
                       L_p * rho_ + L_e * epsilon_

                # Track losses each sample
                epoch_losses = torch.cat((epoch_losses, loss_noisy_labels.data.detach().cpu()))
                epoch_indices = torch.cat((epoch_indices, data_idx.cpu().float()))
                loss.backward()

                # Append predictions
                yhat_hist[data_idx] = yhat_hist[data_idx].roll(1, dims=-1)
                yhat_hist[data_idx, :, 0] = prob.detach()

                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())
                train_acc_corrected.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)

            scheduler.step()
            # Validate
            valid_loss, valid_acc = eval_model(model, valid_data, [loss_ae, loss_class, loss_centroids],
                                               [alpha_, beta_, gamma_])

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            train_acc_epoch = 100 * np.average(train_acc)
            train_acc_corr_epoch = 100 * np.average(train_acc_corrected)

            avg_train_acc.append(train_acc_epoch)
            avg_valid_acc.append(valid_acc)

            print(
                'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - TrAccCorr:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - '
                'ValLoss:{:.5f} - lr:{:.5f} - alpha:{:.3f} - beta:{:.3f} - gamma:{:.3f} - rho:{:.3f} - eps:{:.3f}'
                ' - w_obs:{:.3f} - w_yhat:{:.3f} - w_cen:{:.3f}'
                    .format(idx_epoch, epochs, time() - epochstart, train_acc_epoch, train_acc_corr_epoch,
                            valid_acc, train_loss_epoch, valid_loss, optimizer.param_groups[0]['lr'],
                            alpha_, beta_, gamma_, rho_, epsilon_, w_obs, w_yhat, w_c))

            all_losses.append(epoch_losses)
            all_indices.append(epoch_indices)

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    all_losses = np.vstack(all_losses)
    all_indices = np.vstack(all_indices)

    if plot_loss_flag:
        plot_loss(avg_train_loss, avg_valid_loss, loss_class._get_name(), network, kind='loss', saver=saver,
                  early_stop=0)
        plot_loss(avg_train_acc, avg_valid_acc, loss_class._get_name(), network, kind='accuracy', saver=saver,
                  early_stop=0)

    return model, loss_centroids, (all_losses, all_indices)


def eval_model(model, loader, list_loss, coeffs):
    loss_ae, loss_class, loss_centroids = list_loss
    alpha, beta, gamma = coeffs
    losses = []
    accs = []

    with torch.no_grad():
        model.eval()
        loss_centroids.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            target = Variable(target.long()).to(device)
            batch_size = inputs.size(0)

            out_AE, out_class, embedding = model(inputs)
            ypred = torch.max(F.softmax(out_class, dim=1), dim=1)[1]

            loss_recons_ = loss_ae(out_AE, inputs)
            loss_class_ = loss_class(out_class, target)
            loss_cntrs_ = loss_centroids(embedding.squeeze(), target)
            loss = alpha * loss_recons_ + beta * loss_class_.mean() + gamma * loss_cntrs_.mean()

            losses.append(loss.data.item())

            accs.append((ypred == target).sum().item() / batch_size)

    return np.array(losses).mean(), 100 * np.average(accs)


def train_eval_model(model, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     mask_train, ni, args, saver, correct_labels, plt_embedding=True, plt_loss_hist=True,
                     plt_recons=False, plt_cm=True):
    classes = len(np.unique(Y_train_clean))

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers, pin_memory=True)

    ######################################################################################################
    # Train model
    model, clusterer, (train_losses, train_idxs) = train_model(model, train_loader, valid_loader,
                                                               epochs=args.epochs, args=args, correct=correct_labels,
                                                               saver=saver, plot_loss_flag=args.plt_loss)
    cluster_centers = clusterer.centers.detach().cpu().numpy()
    print('Train ended')

    ######################################################################################################
    # Eval
    train_results = evaluate_class_recons(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                          'CNN', 'Train', correct_labels, plt_cm=plt_cm, plt_lables=False,
                                          plt_recons=plt_recons)
    valid_results = evaluate_class_recons(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                          'CNN', 'Valid', correct_labels, plt_cm=plt_cm, plt_lables=False,
                                          plt_recons=False)
    test_results = evaluate_class_recons(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                         'Test', correct_labels, plt_cm=plt_cm, plt_lables=False, plt_recons=plt_recons)

    if plt_embedding:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, cluster_centers, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network='CNN', saver=saver, correct=correct_labels)

    if ni > 0 and plt_loss_hist:
        visualize_training_loss(train_losses, train_idxs, mask_train, 'CNN', classes, ni, saver,
                                correct=correct_labels)

    plt.close('all')
    torch.cuda.empty_cache()
    return train_results, valid_results, test_results


def main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver):
    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                  norm=args.normalization)

    model_ae = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                     seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                     padding=args.padding, dropout=args.dropout, normalization=args.normalization)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = AEandClass(ae=model_ae, classifier=classifier, name='CNN').to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Validation:', x_valid.shape, Y_valid_clean.shape,
          [(Y_valid_clean == i).sum() for i in np.unique(Y_valid_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])


    ######################################################################################################
    # Main loop
    df_results = pd.DataFrame()
    seeds = np.random.choice(1000, args.n_runs, replace=False)

    args.correct_start = args.init_centers + args.delta_start
    args.correct_end = args.init_centers + args.delta_start + args.delta_end

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
            for correct_labels in args.correct:
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

                train_results, valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test, Y_train,
                                                                              Y_valid, Y_test, Y_train_clean,
                                                                              Y_valid_clean,
                                                                              mask_train, ni, args, saver_slave,
                                                                              correct_labels,
                                                                              plt_embedding=args.plt_embedding,
                                                                              plt_loss_hist=args.plt_loss_hist,
                                                                              plt_recons=args.plt_recons,
                                                                              plt_cm=args.plt_cm)
                if correct_labels:
                    test_corrected_results_main = append_results_dict(test_corrected_results_main, test_results)
                else:
                    test_results_main = append_results_dict(test_results_main, test_results)

                test_results['noise'] = ni
                test_results['noise_type'] = map_losstype(args.label_noise)
                test_results['seed'] = seed
                test_results['correct'] = str(correct_labels)
                test_results['losses'] = map_abg(args.abg)
                test_results['track'] = args.track
                test_results['init_centers'] = args.init_centers
                test_results['delta_start'] = args.delta_start
                test_results['delta_end'] = args.delta_end

                # saver_seed.append_str(['Test Results:'])
                # saver_seed.append_dict(test_results)
                df_results = df_results.append(test_results, ignore_index=True)

            if len(test_results_main):
                keys = list(test_results_main.keys())
            else:
                keys = list(test_corrected_results_main.keys())

        if args.plt_cm:
            fig_title = f"Data:{args.dataset} - Loss:{map_abg(args.abg)} - classes:{classes} - noise:{map_losstype(args.label_noise)}"
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, title=fig_title,
                         x='noise', hue='correct', col=None, kind='bar', style='whitegrid')

    if args.plt_cm:
        fig_title = f"Dataset:{args.dataset} - Loss:{map_abg(args.abg)} - classes:{classes} - noise:{map_losstype(args.label_noise)}"
        plot_results(df_results, keys, saver, title=fig_title,
                     x='noise', hue='correct', col=None, kind='box', style='whitegrid')

    remove_empty_dirs(saver.path)

    return df_results


def single_experiment_SREA(args, path):
    path = os.path.join(path, 'run_{}'.format(args.id))
    saver = SaverSlave(path)

    # Logging setting
    print('run logfile at: ', os.path.join(saver.path, 'logfile.log'))
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
    print('UCR Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)
    print()

    X, Y = load_ucr(args.dataset)
    x_train, x_test, Y_train_clean, Y_test_clean = train_test_split(X, Y, stratify=Y, test_size=0.2)

    Y_valid_clean = Y_test_clean.copy()
    x_valid = x_test.copy()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    batch_size = min(x_train.shape[0] // 10, args.batch_size)
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print('Batch size: ', batch_size)
    args.batch_size = batch_size

    df_results = main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver)
    return df_results
