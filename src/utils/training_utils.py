import os
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from tslearn.metrics import dtw, dtw_path

from src.utils.metrics import *

from time import time
import numpy as np
import matplotlib.pyplot as plt

######################################################################################################
# Global variables

global device, columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print('WARNING: You are using CPU.')
else:
    print('CUDA enabled.')

columns = shutil.get_terminal_size().columns


######################################################################################################

class MapeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(self, input, target):
        return torch.mean(torch.abs(target - input) / (target + self.eps))

class CrossEntropy(nn.Module):
    """Computes the cross-entropy loss
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        # Use log softmax as it has better numerical properties
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.log_softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        return -p


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=True, delta=0, net='mlp', path='./'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.net = net
        self.path = os.path.join(path, net + '_checkpoint.pt')

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def create_dataloader(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size=64, num_workers=1,
                      shuffle_within_batch=True):
    # TODO: update
    # Only shuffle the batches. Keep data inside every batch ordered.
    if not shuffle_within_batch:
        x_train = x_train[:x_train.shape[0] // batch_size * batch_size].reshape(x_train.shape[0] // batch_size,
                                                                                batch_size, *x_train.shape[1:])
        y_train = y_train[:y_train.shape[0] // batch_size * batch_size].reshape(y_train.shape[0] // batch_size,
                                                                                batch_size, *y_train.shape[1:])

        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                                  batch_size=1, shuffle=True,
                                  drop_last=False, num_workers=num_workers)
    else:
        # Shuffle both, batches and data.
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=num_workers)

    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid)),
                              batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=batch_size,
                             shuffle=False,
                             drop_last=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def get_loss_func(loss_type: str, **kwargs):
    loss_type = loss_type.lower()

    loss_functions = {
        'mae': torch.nn.SmoothL1Loss(**kwargs),
        'mse': nn.MSELoss(**kwargs),
        'mape': MapeLoss(**kwargs),
        'bce': torch.nn.BCELoss(**kwargs),
        'crossentropy': torch.nn.CrossEntropyLoss(**kwargs),
        'dilate': DilateLoss(),
        # 'tce': TaylorCrossEntropy(**kwargs)
    }

    if loss_type in loss_functions.keys():
        loss_func = loss_functions[loss_type]
    else:
        raise NameError(f'loss_type {loss_type} not understood. Must be one of {", ".join(loss_functions.keys())}')

    return loss_func


def get_sample_loss(loss_type, **kwargs):
    running_loss_functions = {
        'mae': torch.nn.SmoothL1Loss(reduction='none', **kwargs),
        'mse': nn.MSELoss(reduction='none', **kwargs),
        'bce': torch.nn.BCELoss(reduction='none', **kwargs),
        'crossentropy': torch.nn.CrossEntropyLoss(reduction='none', **kwargs),
        # 'tce': TaylorCrossEntropy(reduction='none', **kwargs)
    }

    if loss_type in running_loss_functions.keys():
        running_loss = running_loss_functions[loss_type]
    else:
        running_loss = None

    return running_loss


def plot_loss(train, valid, loss_type, network, kind='loss', saver=None, early_stop=True, extra_title=''):
    """loss_kind : str : Loss, Accuracy"""
    fig = plt.figure()
    plt.plot(train, label=f'Training {kind}')
    plt.plot(valid, label=f'Validation {kind}')

    if early_stop:
        minposs = valid.index(min(valid))
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.title(f'Model:{network} - Loss:{loss_type} - {extra_title}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if saver is not None:
        # TODO: save in dedicated subfolder
        saver.save_fig(fig, name='{}_training_{}'.format(network, kind), bbox_inches='tight')


def compute_loss(outputs, targets, criteria, ae_loss='mse'):
    if type(outputs) is torch.Tensor:
        # Signle task. Compute the loss
        loss = criteria(outputs, targets)
    elif type(outputs) is list:
        # Multi-task compute same loss for every out
        criteria = [criteria] * len(outputs)
        loss = 0
        for output, target, criterion in zip(outputs, [targets], criteria):
            loss = loss + criterion(output, target)
    elif type(outputs) is tuple:
        criterion_ae = get_loss_func(ae_loss)
        criteria = [criteria] * len(outputs)
        criteria = [criterion_ae] + criteria
        loss = 0
        for output, target, criterion in zip(outputs, targets, criteria):
            loss = loss + criterion(output, target)
    else:
        raise ValueError
    return loss


def train_model_multi(model, loss_type, learning_rate, train_data, valid_data, epochs=100, clip=-1, optimizer=None,
                      scheduler=None, saver=None, patience=0, savepath='./', tensorboard=False,
                      loss_title=None, running_loss=False, ae_loss_type='mse', dict_lossfunc={}):
    network = model.get_name()

    global classification_flag
    if loss_type in ['bce', 'crossentropy', 'tce']:
        classification_flag = True
    else:
        classification_flag = False

    loss_func = get_loss_func(loss_type, **dict_lossfunc)
    ae_loss_func = get_loss_func(ae_loss_type)
    if running_loss:
        running_loss_func = get_sample_loss(loss_type, **dict_lossfunc)
    else:
        running_loss_func = None

    if optimizer == None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if patience > 0:
        early_stopping = EarlyStopping(patience=patience, path=savepath, net=network)

    print('-' * shutil.get_terminal_size().columns)
    print('TRAINING MODEL {} WITH {} LOSS'.format(network, loss_type).center(columns))
    print('-' * shutil.get_terminal_size().columns)

    if tensorboard:
        # TODO: update
        writer_train = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'train'))
        writer_valid = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'valid'))
        print(
            f"tensorboard --logdir={os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network, savepath.split(sep='/')[-1])}")
        # writer.add_graph(model, next(iter(train_data))[0].to(device))
        # writer.close()

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    try:
        running_loss = []
        idx = []
        predictions = []

        for idx_epoch in range(epochs):
            epochstart = time()
            train_loss = []
            train_acc = []

            running_loss_ = []
            idx_ = []
            preds_ = []

            # Train
            model.train()
            opt_step = len(train_data)
            for data, target, data_idx in train_data:
                data = Variable(data.squeeze(0).float()).to(device)
                # Target as long for classification. Float for regression.
                if classification_flag:
                    target = Variable(target.squeeze(0).long()).to(device)
                else:
                    target = Variable(target.squeeze(0).float()).to(device)

                batch_size = data.size(0)

                optimizer.zero_grad()

                out_recons, out_pred = model(data)
                out_pred = out_pred.view_as(target)

                loss_ae = ae_loss_func(out_recons, data)
                loss_class = loss_func(out_pred, target)
                loss = loss_class + loss_ae
                loss.backward()

                if running_loss_func is not None:
                    running_loss_.append(compute_loss(out_pred, target, running_loss_func).data.cpu().numpy().ravel())
                    idx_.append(data_idx.cpu().numpy().ravel())
                    preds_.append(out_pred.data.cpu().numpy())

                # Gradient clip
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())

                if classification_flag:
                    # TODO: multi_out
                    train_acc.append((torch.argmax(out_pred, dim=1) == target).sum().item() / batch_size)

            # LR scheduler
            if scheduler is not None:
                scheduler.step()

            # Sample loss
            if running_loss_func is not None:
                running_loss.append(np.hstack(running_loss_))
                idx.append(np.hstack(idx_))
                predictions.append(np.concatenate(preds_, axis=0))

            # Validate
            valid_loss, valid_acc = eval_model_multi(model, valid_data, [ae_loss_func, loss_func])

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            if classification_flag:
                train_acc_epoch = 100 * np.average(train_acc)
                avg_train_acc.append(train_acc_epoch)
                avg_valid_acc.append(valid_acc)

                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_acc_epoch, valid_acc,
                                train_loss_epoch, valid_loss))
            else:
                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_loss_epoch, valid_loss))

            ## Tensorboard
            if tensorboard:
                # Loss
                writer_train.add_scalar('loss', train_loss, idx_epoch)
                writer_valid.add_scalar('loss', valid_loss, idx_epoch)
                writer_train.add_scalar('learning rate', scheduler.get_lr()[0], idx_epoch)

            # Check early stopping
            if patience > 0:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    if tensorboard:
                        writer_train.close()
                        writer_valid.close()
                    break


    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')
        print('Saving state dict...')
        torch.save(model.state_dict(), os.path.join(savepath, network + '_checkpoint.pt'))

    if saver:
        plot_loss(avg_train_loss, avg_valid_loss, loss_type, network, kind='loss', saver=saver, early_stop=patience,
                  extra_title=loss_title)
        if classification_flag:
            plot_loss(avg_train_acc, avg_valid_acc, loss_type, network, kind='accuracy', saver=saver,
                      early_stop=patience,
                      extra_title=loss_title)

    ## load last checkpoint with best model
    print('Loading best state dict...')
    model.load_state_dict(torch.load(os.path.join(savepath, network + '_checkpoint.pt')))

    print('Save best model')
    torch.save(model, os.path.join(savepath, network + '_model.pt'))

    # Sample loss
    if running_loss:
        running_loss = np.vstack(running_loss)
        idx = np.vstack(idx)
        predictions = np.array(predictions)

    return model, avg_train_loss, avg_valid_loss, (running_loss, predictions, idx)


def train_model(model, loss_type, learning_rate, train_data, valid_data, epochs=100, clip=-1, optimizer=None,
                scheduler=None, saver=None, patience=0, savepath='./', tensorboard=False,
                loss_title=None, running_loss=False, dict_lossfunc={}):
    network = model.get_name()

    global classification_flag
    if loss_type in ['bce', 'crossentropy', 'tce']:
        classification_flag = True
    else:
        classification_flag = False

    loss_func = get_loss_func(loss_type, **dict_lossfunc)
    if running_loss:
        running_loss_func = get_sample_loss(loss_type, **dict_lossfunc)
    else:
        running_loss_func = None

    if optimizer == None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if patience > 0:
        early_stopping = EarlyStopping(patience=patience, path=savepath, net=network)

    print('-' * shutil.get_terminal_size().columns)
    print('TRAINING MODEL {} WITH {} LOSS'.format(network, loss_type).center(columns))
    print('-' * shutil.get_terminal_size().columns)

    if tensorboard:
        # TODO: update
        writer_train = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'train'))
        writer_valid = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'valid'))
        print(
            f"tensorboard --logdir={os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network, savepath.split(sep='/')[-1])}")
        # writer.add_graph(model, next(iter(train_data))[0].to(device))
        # writer.close()

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    try:
        running_loss = []
        idx = []
        for idx_epoch in range(epochs):
            epochstart = time()
            train_loss = []
            train_acc = []

            running_loss_ = []
            idx_ = []

            # Train
            model.train()
            opt_step = len(train_data)
            for data, target, data_idx in train_data:
                data = Variable(data.squeeze(0).float()).to(device)
                # Target as long for classification. Float for regression.
                if classification_flag:
                    target = Variable(target.squeeze(0).long()).to(device)
                else:
                    target = Variable(target.squeeze(0).float()).to(device)

                batch_size = data.size(0)

                optimizer.zero_grad()

                output = model(data)

                loss = compute_loss(output, target, loss_func)
                loss.backward()

                if running_loss_func is not None:
                    running_loss_.append(compute_loss(output, target, running_loss_func).data.cpu().numpy().ravel())
                    idx_.append(data_idx.cpu().numpy().ravel())

                # Gradient clip
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())

                if classification_flag:
                    # TODO: multi_out
                    train_acc.append((torch.argmax(output, dim=1) == target).sum().item() / batch_size)

            # LR scheduler
            if scheduler is not None:
                scheduler.step()

            # Sample loss
            if running_loss_func is not None:
                running_loss.append(np.hstack(running_loss_))
                idx.append(np.hstack(idx_))

            # Validate
            valid_loss, valid_acc = eval_model(model, valid_data, loss_func)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            if classification_flag:
                train_acc_epoch = 100 * np.average(train_acc)
                avg_train_acc.append(train_acc_epoch)
                avg_valid_acc.append(valid_acc)

                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_acc_epoch, valid_acc,
                                train_loss_epoch, valid_loss))
            else:
                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_loss_epoch, valid_loss))

            ## Tensorboard
            if tensorboard:
                # Loss
                writer_train.add_scalar('loss', train_loss, idx_epoch)
                writer_valid.add_scalar('loss', valid_loss, idx_epoch)
                writer_train.add_scalar('learning rate', scheduler.get_lr()[0], idx_epoch)

            # Check early stopping
            if patience > 0:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    if tensorboard:
                        writer_train.close()
                        writer_valid.close()
                    break


    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')
        print('Saving state dict...')
        torch.save(model.state_dict(), os.path.join(savepath, network + '_checkpoint.pt'))

    if saver:
        plot_loss(avg_train_loss, avg_valid_loss, loss_type, network, kind='loss', saver=saver, early_stop=patience,
                  extra_title=loss_title)
        if classification_flag:
            plot_loss(avg_train_acc, avg_valid_acc, loss_type, network, kind='accuracy', saver=saver,
                      early_stop=patience,
                      extra_title=loss_title)

    ## load last checkpoint with best model
    print('Loading best state dict...')
    model.load_state_dict(torch.load(os.path.join(savepath, network + '_checkpoint.pt')))

    print('Save best model')
    torch.save(model, os.path.join(savepath, network + '_model.pt'))

    # Sample loss
    if running_loss:
        running_loss = np.vstack(running_loss)
        idx = np.vstack(idx)

    return model, avg_train_loss, avg_valid_loss, (running_loss, idx)


def old_train_model(model, loss_type, learning_rate, train_data, valid_data, epochs=100, clip=4.0, optimizer=None,
                    scheduler=None, saver=None, patience=5, savepath='./', network='MLP', loss_weight=None,
                    tensorboard=False, TCEorder=2):
    global classification_flag
    if loss_type.lower() in ['bce', 'crossentropy', 'tce']:
        classification_flag = True
    else:
        classification_flag = False

    print('TRAINING MODEL {} WITH {} LOSS'.format(network, loss_type))

    if loss_type.lower() == 'mae':
        loss_func = torch.nn.SmoothL1Loss()
    elif loss_type.lower() == 'mse':
        loss_func = nn.MSELoss(reduction='mean')
        loss_func_running = nn.MSELoss(reduction='none')
    elif loss_type.lower() == 'mape':
        loss_func = MapeLoss()
    elif loss_type.lower() == 'bce':
        loss_func = torch.nn.BCELoss()
    elif loss_type.lower() == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(weight=loss_weight)
        loss_func_running = torch.nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
    elif loss_type.lower() == 'dilate':
        # TODO: optimize this
        loss_func = DilateLoss()
    elif loss_type.lower() == 'tce':
        loss_func = TaylorCrossEntropy(order=TCEorder)
    else:
        raise ValueError('define a loss_type (case insensitive)')

    if optimizer == None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience, path=savepath, net=network)

    if tensorboard:
        writer_train = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'train'))
        writer_valid = SummaryWriter(
            log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network,
                                 savepath.split(sep='/')[-1], 'valid'))
        print(
            f"tensorboard --logdir={os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network, savepath.split(sep='/')[-1])}")
        # writer.add_graph(model, next(iter(train_data))[0].to(device))
        # writer.close()

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    try:
        running_loss = []
        idx = []
        for idx_epoch in range(epochs):
            epochstart = time()
            train_loss = []
            train_acc = []

            running_loss_ = []
            idx_ = []

            # Train
            model.train()
            opt_step = len(train_data)
            for data, target, data_idx in train_data:
                data = Variable(data.squeeze(0).float()).to(device)
                if classification_flag:
                    target = Variable(target.squeeze(0).long()).to(device)
                else:
                    target = Variable(target.squeeze(0).float()).to(device)
                batch_size = data.size(0)

                optimizer.zero_grad()

                output = model(data)

                loss = loss_func(output, target)

                loss.backward()

                running_loss_.append(loss_func_running(output, target).data.cpu().numpy().ravel())
                idx_.append(data_idx.cpu().numpy().ravel())

                """# VISUALIZE GRADIENTS IN TENSORBOARD DURING TRAINING
                gradmean = []
                gradmax = []
                names = []
                # Gradients
                for tag, parm in model.named_parameters():
                    # Old plotter
                    # writer_train.add_histogram(tag, parm.grad.data.cpu().numpy(), iter)

                    gradmean.append(np.abs(parm.grad.clone().detach().cpu().numpy()).mean())
                    gradmax.append(np.max(parm.grad.clone().detach().cpu().numpy()))
                    names.append(tag)
                gradmean = np.vstack(gradmean)
                gradmax = np.vstack(gradmax)

                _limits = np.array([float(i) for i in range(len(gradmean))])
                _num = len(gradmean)
                writer_train.add_histogram_raw(tag=network + "/abs_mean", min=0.0, max=0.3, num=_num,
                                               sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(),
                                               bucket_limits=_limits,
                                               bucket_counts=gradmean, global_step=iter)

                writer_train.add_histogram_raw(tag=network + "/signed_max", min=0.0, max=0.3, num=_num,
                                               sum=gradmax.sum(), sum_squares=np.power(gradmax, 2).sum(),
                                               bucket_limits=_limits,
                                               bucket_counts=gradmax, global_step=iter)

                _mean = {}
                _max = {}
                for i, name in enumerate(names):
                    _mean[name] = gradmean[i]
                    _max[name] = gradmax[i]

                writer_train.add_scalars(network + "/abs_mean", _mean, global_step=iter)
                writer_train.add_scalars(network + "/signed_max", _max, global_step=iter)"""

                # Gradient clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Update pbar
                train_loss.append(loss.data.item())

                if classification_flag:
                    train_acc.append((torch.argmax(output, dim=1) == target).sum().item() / batch_size)

            # LR scheduler
            if scheduler is not None:
                scheduler.step()

            # Sample loss
            running_loss.append(np.hstack(running_loss_))
            idx.append(np.hstack(idx_))

            # Validate
            valid_loss, valid_acc = eval_model(model, valid_data, loss_func)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            if classification_flag:
                train_acc_epoch = 100 * np.average(train_acc)
                avg_train_acc.append(train_acc_epoch)
                avg_valid_acc.append(valid_acc)

                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_acc_epoch, valid_acc,
                                train_loss_epoch, valid_loss))
            else:
                print(
                    'Epoch [{}/{}], Time:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f}'
                        .format(idx_epoch + 1, epochs, time() - epochstart, train_loss_epoch, valid_loss))

            ## Tensorboard
            if tensorboard:
                # Loss
                writer_train.add_scalar('loss', train_loss, idx_epoch)
                writer_valid.add_scalar('loss', valid_loss, idx_epoch)
                writer_train.add_scalar('learning rate', scheduler.get_lr()[0], idx_epoch)

            # Check early stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                if tensorboard:
                    writer_train.close()
                    writer_valid.close()
                break


    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')
        print('Saving model...')
        torch.save(model.state_dict(), os.path.join(savepath, network + '_checkpoint.pt'))

    fig = plt.figure()
    plt.plot(avg_train_loss, label='Training Loss')
    plt.plot(avg_valid_loss, label='Validation Loss')
    minposs = avg_valid_loss.index(min(avg_valid_loss))
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.title(f'Network: {network} Loss: {loss_type}')
    plt.grid(True)
    plt.legend(loc=1)
    plt.tight_layout()
    # plt.show()
    if saver is not None:
        saver.save_fig(fig, name='{:}_training_loss'.format(network), bbox_inches='tight')

    if classification_flag:
        fig = plt.figure()
        plt.plot(avg_train_acc, label='Training Accuracy')
        plt.plot(avg_valid_acc, label='Validation Accuracy')
        minposs = avg_valid_acc.index(max(avg_valid_acc))
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title(f'Network: {network} Loss: {loss_type}')
        plt.grid(True)
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show()
        if saver is not None:
            saver.save_fig(fig, name='{:}_training_acc'.format(network), bbox_inches='tight')

    ## load last checkpoint with best model
    print('Loading best model...')
    model.load_state_dict(torch.load(os.path.join(savepath, network + '_checkpoint.pt')))

    print('Save best model')
    torch.save(model, os.path.join(savepath, network + '_model.pt'))

    # Sample loss
    running_loss = np.vstack(running_loss)
    idx = np.vstack(idx)

    return model, avg_train_loss, avg_valid_loss, (running_loss, idx)


def eval_model(model, loader, loss_func):
    losses = []
    acc = []

    with torch.no_grad():
        model.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            batch_size = inputs.size(0)
            if classification_flag:
                target = Variable(target.long()).to(device)
            else:
                target = Variable(target.float()).to(device)

            outputs = model(inputs)

            loss = compute_loss(outputs, target, loss_func)

            losses.append(loss.data.item())

            if classification_flag:
                # TODO: multi task
                acc.append((torch.argmax(outputs, dim=1) == target).sum().item() / batch_size)

    if classification_flag:
        return np.array(losses).mean(), 100 * np.average(acc)
    else:
        return np.array(losses).mean(), 0


def eval_model_multi(model, loader, loss_func):
    losses = []
    acc = []

    with torch.no_grad():
        model.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            batch_size = inputs.size(0)
            if classification_flag:
                target = Variable(target.long()).to(device)
            else:
                target = Variable(target.float()).to(device)

            out_recons, out_pred = model(inputs)
            out_pred = out_pred.view_as(target)

            loss_ae = loss_func[0](out_recons, inputs)
            loss_class = loss_func[1](out_pred, target)
            loss = loss_class + loss_ae
            losses.append(loss.data.item())

            if classification_flag:
                # TODO: multi task
                acc.append((torch.argmax(out_pred, dim=1) == target).sum().item() / batch_size)

    if classification_flag:
        return np.array(losses).mean(), 100 * np.average(acc)
    else:
        return np.array(losses).mean(), 0





def compute_metrics(y_hat, y):
    metrics = dict()
    metrics['mae'] = np.mean(np.abs(y - y_hat))
    metrics['mse'] = np.mean(np.square(y - y_hat))
    metrics['rmse'] = np.mean(np.sqrt(np.square(y - y_hat)))
    metrics['mape'] = mape(y, y_hat)
    metrics['smape'] = smape(y, y_hat)
    metrics['mase'] = mase(y, y_hat)
    metrics['rae'] = rae(y, y_hat)
    metrics['mrae'] = mrae(y, y_hat)
    # metrics['dtw'] = dtw_metric(y_hat, y)
    # metrics['tdi'] = tdi_metric(y_hat, y)

    s = 'Score: '
    for key, value in metrics.items():
        s += f'{key}:{value} - '
    print(s)
    return metrics


def train_eval_func(model, train_loader, valid_loader, test_loader, y_test, saver, network,
                    LOSS, LR, EPOCHS, PATIENCE, MET):
    results = dict()
    # Run all the loss function
    for loss_type in LOSS:
        model, _, _ = train_model(model, loss_type, LR, train_loader, valid_loader, epochs=EPOCHS, saver=saver,
                                  patience=PATIENCE, savepath=saver.path, network=network)
        y_hat = predict(model, test_loader)
        # results[loss_type] = compute_metrics(y_hat, y_test)
        results[loss_type] = evaluate(actual=y_test, predicted=y_hat, metrics=MET)

        s = '{:}({:})'.format(network, loss_type)
        for k, v in zip(results[loss_type].keys(), results[loss_type].values()):
            s += '\t{:}:{:.4f}'.format(k, v)
        saver.append_str([s])

        try:
            if y_test.shape[1] > 1:
                plot_prediction(y_test, y_hat, title=s.replace('\t', ' '), saver=saver,
                                figname='{:}_{:}'.format(network, loss_type))
        except IndexError:
            pass
        plot_test(y_test, y_hat, s.replace('\t', ' '), saver=saver)
        plt.close('all')
        print()
    saver.append_str([''])
    return results


def train_model_checkpoints(model, loss_type, learning_rate, train_data, valid_data, epochs=100, clip=4.0,
                            optimizer=None, scheduler=None, savepath='./', network='MLP', saver=None, loss_weight=None):
    print('TRAINING MODEL {} WITH {} LOSS'.format(network, loss_type))

    if loss_type.lower() == 'mae':
        loss_func = torch.nn.SmoothL1Loss()
    elif loss_type.lower() == 'mse':
        loss_func = nn.MSELoss(reduction='mean')
    elif loss_type.lower() == 'mape':
        loss_func = MapeLoss()
    elif loss_type.lower() == 'bce':
        loss_func = torch.nn.BCELoss()
    elif loss_type.lower() == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif loss_type.lower() == 'dilate':
        # TODO: optimize this
        loss_func = DilateLoss()
    else:
        raise ValueError('define a loss_type (case insensitive)')

    if optimizer == None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # writer = SummaryWriter(
    #    log_dir=os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network, savepath.split(sep='/')[-1]))

    # print(
    #    f"tensorboard --logdir={os.path.join(savepath.split(sep='results')[0], 'results', 'runs', network, savepath.split(sep='/')[-1])}")
    # writer.add_graph(model, next(iter(train_data))[0].to(device))
    # writer.close()

    epoch_idx = 1
    print('Save checkpoint at 0% of training')
    torch.save(model.state_dict(),
               os.path.join(savepath, network + '_checkpoint0.pt'))

    avg_train_loss = []
    avg_valid_loss = []

    try:
        # Iteration counter, needed to visualize gradients
        iter = 0
        for epoch in range(epochs):
            epochstart = time()
            train_loss = []

            # Train
            model.train()
            for data, target in train_data:
                data = Variable(data.squeeze(0).float()).to(device)
                target = Variable(target.squeeze(0).long()).to(device)

                optimizer.zero_grad()

                output = model(data)

                loss = loss_func(output, target)
                train_loss.append(loss.data.item())

                loss.backward()

                # Gradient clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                """# VISUALIZE GRADIENTS IN TENSORBOARD DURING TRAINING
                gradmean = []
                gradmax = []
                names = []
                # Gradients
                for tag, parm in model.named_parameters():
                    # Old plotter
                    # writer_train.add_histogram(tag, parm.grad.data.cpu().numpy(), iter)

                    gradmean.append(np.abs(parm.grad.clone().detach().cpu().numpy()).mean())
                    gradmax.append(np.max(parm.grad.clone().detach().cpu().numpy()))
                    names.append(tag)
                gradmean = np.vstack(gradmean)
                gradmax = np.vstack(gradmax)

                _limits = np.array([float(i) for i in range(len(gradmean))])
                _num = len(gradmean)
                writer.add_histogram_raw(tag=network + "/abs_mean", min=0.0, max=0.3, num=_num,
                                         sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(),
                                         bucket_limits=_limits,
                                         bucket_counts=gradmean, global_step=iter)

                writer.add_histogram_raw(tag=network + "/signed_max", min=0.0, max=0.3, num=_num,
                                         sum=gradmax.sum(), sum_squares=np.power(gradmax, 2).sum(),
                                         bucket_limits=_limits,
                                         bucket_counts=gradmax, global_step=iter)

                _mean = {}
                _max = {}
                for i, name in enumerate(names):
                    _mean[name] = gradmean[i]
                    _max[name] = gradmax[i]

                writer.add_scalars(network + "/abs_mean", _mean, global_step=iter)
                writer.add_scalars(network + "/signed_max", _max, global_step=iter)"""

                # Increase iterator counter
                iter += 1

            # LR scheduler
            if scheduler is not None:
                scheduler.step()
                # writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)

            # Validate
            valid_loss = eval_model(model, valid_data, loss_func)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            # Tensorboard
            _loss = {}
            _loss['train'] = train_loss_epoch
            _loss['valid'] = valid_loss
            # writer.add_scalars('losses', _loss, global_step=epoch)

            print(
                'Epoch [{}/{}], Time:{:.3f} - train_loss:{:.5f} - valid_loss:{:.5f}'
                    .format(epoch + 1, epochs, time() - epochstart, train_loss_epoch, valid_loss))

            # Every epoch
            print('Save checkpoint at {}/{} epoch'.format(epoch + 1, epochs))
            torch.save(model.state_dict(),
                       os.path.join(savepath, network + '_checkpoint{}.pt'.format(epoch + 1)))
        # writer.close()

    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')

    if saver is not None:
        fig = plt.figure()
        plt.plot(avg_train_loss, label='Training Loss')
        plt.plot(avg_valid_loss, label='Validation Loss')
        minposs = avg_valid_loss.index(min(avg_valid_loss))
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title(f'Network: {network} Loss: {loss_type}')
        plt.grid(True)
        plt.legend(loc=1)
        plt.tight_layout()
        # plt.show()
        saver.save_fig(fig, name='{:}_training_loss'.format(network), bbox_inches='tight')

    print('Save model')
    torch.save(model, os.path.join(savepath, network + '_model.pt'))

    return model, avg_train_loss, avg_valid_loss
