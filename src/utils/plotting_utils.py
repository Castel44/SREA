import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D

import src.utils.utils as utils
from src.utils.decorators import repeat, reset_rng


######################################################################################################
class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


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


def plot_test_reuslts(test: dict, test_correct: dict, ni_list: list, classes: int, network: str, seed: int,
                      saver: object, abg=None) -> None:
    if test.keys() != test_correct.keys():
        print('Plain and Corrected dict_keys are different. Plotting only test...')
        test = {k: [] for k in test_correct.keys()}
    if abg == None:
        abg = utils.map_abg([1, 1, 1])

    n = len(test.keys())

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, 5 + (n * 0.1)), sharex='all')
    for ax, (key, tst), tst_corr in zip(axes, test.items(), test_correct.values()):
        ax.plot(tst, '--o', label='Test (Naive)')
        ax.plot(tst_corr, '--s', label='Test (Proposed)')
        ax.set_ylim([0, 1])

        ax.set_ylabel('{}'.format(key))
        ax.set_xticks([i for i in range(len(ni_list))])
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[-1].set_xticklabels(ni_list)
    axes[-1].set_xlabel('Label Noise ratio')
    axes[0].set_title('Model:{} - n_classes:{} - seed:{} - L:{}'.format(network, classes, seed, abg))
    fig.tight_layout()

    saver.save_fig(fig)


def boxplot_results(data, keys, classes, network, saver):
    n = len(keys)
    x = 'noise'
    hue = 'correct'
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 7 + (n * 0.1)), sharex='all')
    for ax, k in zip(axes, keys):
        sns.boxplot(x=x, y=k, hue=hue, data=data, ax=ax)
        ax.grid(True, alpha=0.2)
    axes[0].set_title('Model:{} classes:{}'.format(network, classes))
    fig.tight_layout()
    saver.save_fig(fig, 'boxplot')


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


def plot_embedding(model, train_loader, valid_loader, cluster_centers, Y_train_clean, Y_valid_clean, Y_train, Y_valid,
                   saver, network='Model', correct=False):
    print('Plot Embedding...')
    # Embeddings
    train_embedding = utils.predict(model, train_loader).squeeze()
    valid_embedding = utils.predict(model, valid_loader).squeeze()
    centroids_embedding = cluster_centers
    classes = len(np.unique(Y_train_clean))

    ttl = f'{network} - Embedding'
    n_comp = 2
    if train_embedding.shape[-1] > 3:
        from umap import UMAP
        trs = UMAP(n_components=n_comp, n_neighbors=50, min_dist=0.01, metric='euclidean')
        ttl = 'UMAP'
        train_embedding2d = trs.fit_transform(train_embedding)
        valid_embedding2d = trs.transform(valid_embedding)
        centroids = trs.transform(centroids_embedding)
    else:
        train_embedding2d = train_embedding
        valid_embedding2d = valid_embedding
        centroids = centroids_embedding

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
    l2 = ax.scatter(*centroids.T, s=250, marker='P', label='Learnt Centroids',
                    c=COL.get_rgb([i for i in range(classes)]), edgecolors='black')
    lines = [l0, l1, l2] + [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=10) for c in
                            [COL.get_rgb(i) for i in np.unique(Y_train_clean.astype(int))]]
    labels = [l0.get_label(), l1.get_label(), l2.get_label()] + [i for i in range(len(lines))]
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

    # plot_hists_ephocs(train_losses, mask_train, auc=False, nrows=3, ncols=3, net=network, classes=classes,
    #                  saver=saver, ni=ni, pred_ni=0)

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


@reset_rng
@repeat(num_times=1)
def plot_prediction(X, X_hat, nrows=5, ncols=5, figsize=(19.2, 10), title: str = 'model', saver: object = None,
                    figname: str = ''):
    # TODO: export indices to havve a fair comparison across different methods
    # Setting seed for reproducibility.
    idx = np.random.randint(0, X_hat.shape[0], nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight=600)

    for i, ax in enumerate(axes.ravel()):
        ax.plot(X[idx[i]], '--', label='Original')
        ax.plot(X_hat[idx[i]], label='Recons')
        # ax.set_ylim([-0.05, 1.05])
        # if i == ncols//2:
        #    ax.set_title(title, fontsize=12, fontweight=600)

    plt.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    # plt.show()

    if saver:
        saver.save_fig(fig, figname + '_pred')
