import argparse
import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.switch_backend('Qt5Agg')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style('whitegrid')


def drop_constant_column(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]


path_list_symm = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Symm/1_1_1/2021-03-14T11:24:44/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Symm/1_1_1/2021-03-14T11:24:41/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Symm/1_1_1/2021-03-14T11:24:52/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Symm/1_1_1/2021-03-14T11:25:17/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Symm/1_1_1/2021-03-14T11:25:26/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Symm/1_1_1/2021-03-14T11:25:34/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Symm/1_1_1/2021-03-14T11:25:02/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Symm/1_1_1/2021-03-14T11:17:51/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Symm/1_1_1/2021-03-14T11:25:14/',
    # MELBOURNE
]

path_list_asymm = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Asymm_1/1_1_1/2021-03-14T17:46:52/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Asymm_1/1_1_1/2021-03-14T18:16:14/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Asymm_1/1_1_1/2021-03-14T14:51:01/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Asymm_1/1_1_1/2021-03-14T16:24:25/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Asymm_1/1_1_1/2021-03-14T16:32:54/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Asymm_1/1_1_1/2021-03-14T16:53:30/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Asymm_1/1_1_1/2021-03-14T15:56:45/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Asymm_1/1_1_1/2021-03-14T15:07:54/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Asymm_1/1_1_1/2021-03-14T14:28:49/',
    # MELBOURNE
]

out_path_ = '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/results/'

for noise_type, path_list in zip(['symmetric', 'asymm_1'], [path_list_symm, path_list_asymm]):
    out_path = os.path.join(out_path_, noise_type)
    os.makedirs(out_path, exist_ok=True)

    df_init_centers = pd.DataFrame()
    df_start = pd.DataFrame()
    df_end = pd.DataFrame()

    for path in path_list:
        csv = None
        for file in os.listdir(path):
            if file.endswith('.csv'):
                print('DataFrame found: {}'.format(file))
                csv = file
                break
        if csv is None:
            raise ValueError

        df = pd.read_csv(os.path.join(path, file))

        df.rename(columns={'init_centers': r'$\lambda_{init}$',
                           'delta_end': r'$\Delta_{end}$',
                           'delta_start': r'$\Delta_{start}$'}, inplace=True)

        # Rank per dataset
        # df['acc_rank'] = df.groupby(['noise'])['acc'].rank(method='max', ascending=True, pct=True)
        df['f1_weighted_rank'] = df.groupby(['noise'])['f1_weighted'].rank(method='average', ascending=True, pct=True)

        method = 'average'
        a = df.groupby(['noise', r'$\lambda_{init}$'])['f1_weighted_rank'].mean()
        df_init_centers = df_init_centers.append(a.reset_index())
        a = df.groupby(['noise', r'$\Delta_{start}$'])['f1_weighted_rank'].mean()
        df_start = df_start.append(a.reset_index())
        a = df.groupby(['noise', r'$\Delta_{end}$'])['f1_weighted_rank'].mean()
        df_end = df_end.append(a.reset_index())

    os.makedirs(out_path, exist_ok=True)

    g = sns.catplot(data=df_init_centers, y='f1_weighted_rank', x='noise', kind='bar', hue=r'$\lambda_{init}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('init_centers rank')
    plt.savefig(os.path.join(out_path, 'init_centers_rank_bar.png'))
    g = sns.catplot(data=df_init_centers, y='f1_weighted_rank', kind='bar', x=r'$\lambda_{init}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('init_centers rank')
    plt.savefig(os.path.join(out_path, 'init_centers_rank.png'))
    # g = sns.catplot(data=df_init_centers, y='f1_weighted_rank', row='noise', kind='box', x=r'$\lambda_{init}$')
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle('init_centers rank')
    # plt.savefig(os.path.join(out_path, 'init_centers_rank_box.png'))

    g = sns.catplot(data=df_start, y='f1_weighted_rank', x='noise', kind='bar', hue=r'$\Delta_{start}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('delta_start rank')
    plt.savefig(os.path.join(out_path, 'delta_start_rank_bar.png'))
    g = sns.catplot(data=df_start, y='f1_weighted_rank', kind='bar', x=r'$\Delta_{start}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('delta_start rank')
    plt.savefig(os.path.join(out_path, 'delta_start.png'))
    # g = sns.catplot(data=df_start, y='f1_weighted_rank', row='noise', kind='box', x=r'$\Delta_{start}$')
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle('delta_start rank')
    # plt.savefig(os.path.join(out_path, 'delta_start_rank_box.png'))

    g = sns.catplot(data=df_end, y='f1_weighted_rank', x='noise', kind='bar', hue=r'$\Delta_{end}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('delta_end rank')
    plt.savefig(os.path.join(out_path, 'delta_end_rank_bar.png'))
    g = sns.catplot(data=df_end, y='f1_weighted_rank', kind='bar', x=r'$\Delta_{end}$', errwidth=0.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('delta_end rank')
    plt.savefig(os.path.join(out_path, 'delta_end_rank.png'))
    # g = sns.catplot(data=df_end, y='f1_weighted_rank', row='noise', kind='box', x=r'$\Delta_{end}$')
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle('delta_end rank')
    # plt.savefig(os.path.join(out_path, 'delta_end_rank_box.png'))

    print('results in:', out_path)
    print()
