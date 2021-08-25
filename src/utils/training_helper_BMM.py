import collections
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import scipy.stats as stats

from src.models.model import CNNAE
from src.models.MultiTaskClassification import MetaModel, LinClassifier, NonLinClassifier
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class
from src.utils.plotting_utils import plot_results, plot_embedding

######################################################################################################
columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################################################
##################### Loss tracking and noise modeling #######################


def track_training_loss(args, model, device, train_loader, epoch, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)

        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, target, reduction='none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr <= max_perc) & (loss_tr >= min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6

    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (
                bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr >= 1] = 1 - 10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)

    return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_maxLoss, bmm_model_minLoss


##############################################################################

########################### Cross-entropy loss ###############################
def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        output = F.log_softmax(output, dim=1)

        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################

############################# Mixup original #################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)


def train_mixUp(args, model, device, train_loader, optimizer, epoch, alpha):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)

        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        loss = mixup_criterion(output, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################

########################## Mixup + Dynamic Hard Bootstrapping ##################################
# Mixup with hard bootstrapping using the beta model
def reg_loss_class(mean_tab, num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1. / num_classes) * torch.log((1. / num_classes) / items)
    return loss


def mixup_data_Boot(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device == 'cuda':
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, \
                             bmm_model_maxLoss, bmm_model_minLoss, reg_term, num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        # output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean, -2)
        output = F.log_softmax(output, dim=1)

        B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)

        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam * (loss_x1 + loss_x1_pred) + (1 - lam) * (loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term * loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################


##################### Mixup Beta Soft Bootstrapping ####################
# Mixup guided by our beta model with beta soft bootstrapping

def mixup_criterion_mixSoft(pred, y_a, y_b, B, lam, index, output_x1, output_x2):
    return torch.sum(
        (lam) * (
                (1 - B) * F.nll_loss(pred, y_a, reduction='none') + B * (
            -torch.sum(F.softmax(output_x1, dim=1) * pred, dim=1))) +
        (1 - lam) * (
                (1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + B[index] * (
            -torch.sum(F.softmax(output_x2, dim=1) * pred, dim=1)))) / len(
        pred)


def train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, bmm_model_maxLoss, \
                             bmm_model_minLoss, reg_term, num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1 = output_x1.detach()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5 * torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]

        tab_mean_class = torch.mean(output_mean, -2)  # Columns mean

        loss = mixup_criterion_mixSoft(output, targets_1, targets_2, B, lam, index, output_x1,
                                       output_x2)
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term * loss_reg
        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################

################################ Dynamic Mixup ##################################
# Mixup guided by our beta model

def mixup_data_beta(x, y, B, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if device == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    lam = ((1 - B) + (1 - B[index]))

    fac1 = ((1 - B) / lam)
    fac2 = ((1 - B[index]) / lam)
    for _ in range(x.dim() - 1):
        fac1.unsqueeze_(1)
        fac2.unsqueeze_(1)
    mixed_x = fac1 * x + fac2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index


def mixup_criterion_beta(pred, y_a, y_b):
    lam = np.random.beta(32, 32)
    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)


def train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,
                     bmm_model_maxLoss, bmm_model_minLoss):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5 * torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
        output = model(inputs_mixed)
        output = F.log_softmax(output, dim=1)

        loss = mixup_criterion_beta(output, targets_1, targets_2)

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


################################################################################


################## Dynamic Mixup + soft2hard bootstraping ##################
def mixup_criterion_SoftHard(pred, y_a, y_b, B, index, output_x1, output_x2, Temp):
    return torch.sum(
        (0.5) * (
                (1 - B) * F.nll_loss(pred, y_a, reduction='none') + B * (
            -torch.sum(F.softmax(output_x1 / Temp, dim=1) * pred, dim=1))) +
        (0.5) * (
                (1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + B[index] * (
            -torch.sum(F.softmax(output_x2 / Temp, dim=1) * pred, dim=1)))) / len(
        pred)


def train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, epoch, bmm_model, \
                                   bmm_model_maxLoss, bmm_model_minLoss, countTemp, k, temp_length, reg_term,
                                   num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    steps_every_n = 2  # 2 means that every epoch we change the value of k (index)
    temp_vec = np.linspace(1, 0.001, temp_length)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1 = output_x1.detach()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5 * torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]
        tab_mean_class = torch.mean(output_mean, -2)

        Temp = temp_vec[k]

        loss = mixup_criterion_SoftHard(output, targets_1, targets_2, B, index, output_x1, output_x2, Temp)
        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term * loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, Temperature: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * args.batch_size),
                    optimizer.param_groups[0]['lr'], Temp))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    countTemp = countTemp + 1
    if countTemp == steps_every_n:
        k = k + 1
        k = min(k, len(temp_vec) - 1)
        countTemp = 1

    return (loss_per_epoch, acc_train_per_epoch, countTemp, k)


########################################################################


def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    cnn_model.eval()
    outputs = cnn_model(data)
    outputs = F.log_softmax(outputs, dim=1)
    batch_losses = F.nll_loss(outputs.float(), target, reduction='none')
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1 - 10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    # B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)


def test_cleaning(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    # acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


def compute_loss_set(args, model, device, data_loader):
    model.eval()
    all_losses = torch.Tensor()
    for batch_idx, (data, target) in enumerate(data_loader):
        prediction = model(data.to(device))
        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction.float(), target.to(device), reduction='none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
    return all_losses.data.numpy()


def val_cleaning(args, model, device, val_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * args.val_batch_size))

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.average(acc_val_per_batch)]
    return (loss_per_epoch, acc_val_per_epoch)


################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


#######################################################################################################################
## HELPER FUNCTION CODE
#######################################################################################################################

def train_model(model, train_loader, valid_loader, test_loader, mixup, bootbeta, args, saver=None):
    network = model.get_name()
    milestones = args.M
    num_classes = args.nbins

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} MixUp - {} BootBeta'.format(network, mixup, bootbeta).center(
        columns)
    print(s)
    print('-' * shutil.get_terminal_size().columns)
    # saver.append_str(['*' * 100, s, '*' * 100])

    optimizer = torch.optim.Adam(
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
    train_results = evaluate_class(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                          'CNN', 'Train', correct_labels, plt_cm=plt_cm, plt_lables=False)
    valid_results = evaluate_class(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                          'CNN', 'Valid', correct_labels, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                         'Test', correct_labels, plt_cm=plt_cm, plt_lables=False)

    if plt_embedding and args.embedding_size <= 3:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network='CNN', saver=saver, correct=correct_labels)

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

    model_ae = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                     seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                     padding=args.padding, dropout=args.dropout, normalization=args.normalization)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel(ae=model_ae, classifier=classifier, name='CNN').to(device)
    # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

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
            for correct_labels in args.correct:
                i += 1
                # True or false
                print('+' * shutil.get_terminal_size().columns)
                print('HyperRun: %d/%d' % (i, len(args.ni) * len(args.correct)))
                print('Label noise ratio: %.3f' % ni)
                print('Correct labels:', correct_labels)
                print('+' * shutil.get_terminal_size().columns)
                # saver.append_str(['#' * 100, 'Label noise ratio: %f' % ni, 'Correct Labels: %s' % correct_labels])

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
                test_results['noise'] = ni * .01
                test_results['seed'] = seed
                test_results['correct'] = str(correct_labels)
                test_results['losses'] = map_abg([0, 1, 0])
                df_results = df_results.append(test_results, ignore_index=True)

        if args.plt_cm:
            fig_title = f"Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} " \
                        f"- MixUp:{args.Mixup} - BootBeta:{args.BootBeta}"
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, x='noise', hue='correct',
                         col='losses',
                         kind='bar', style='whitegrid', title=fig_title)

    if args.plt_cm:
        # Losses column should  not change here
        fig_title = f"Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} " \
                    f"- MixUp:{args.Mixup} - BootBeta:{args.BootBeta}"
        plot_results(df_results, keys, saver, x='noise', hue='correct', col='losses', kind='box', style='whitegrid',
                     title=fig_title)

    # boxplot_results(df_results, keys, classes, 'CNN', args.Mixup, args.BootBeta, saver)

    # results_summary = df_results.groupby(['noise', 'correct'])[keys].describe().T
    # saver.append_str(['Results main summary', results_summary])

    remove_empty_dirs(saver.path)

    return df_results
