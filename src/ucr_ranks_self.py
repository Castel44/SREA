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
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Symm/Variable/2021-03-14T20:20:40/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Symm/Variable/2021-03-15T01:35:15/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Symm/Variable/2021-03-15T01:35:40/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Symm/Variable/2021-03-15T01:36:30/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Symm/Variable/2021-03-15T01:36:59/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Symm/Variable/2021-03-15T01:32:58/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Symm/Variable/2021-03-15T12:55:48/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/MelbournePedestrian/CNN/Symm/Variable/2021-03-15T10:17:49/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Symm/Variable/2021-03-15T12:59:38/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Symm/Variable/2021-03-15T13:00:48/'

]

path_list_asymm = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Asymm_1/Variable/2021-03-14T21:28:27/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Asymm_1/Variable/2021-03-15T02:58:43/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Asymm_1/Variable/2021-03-15T04:22:57/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Asymm_1/Variable/2021-03-15T04:37:06/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Asymm_1/Variable/2021-03-15T03:42:53/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Asymm_1/Variable/2021-03-15T03:51:02/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Asymm_1/Variable/2021-03-15T15:54:11/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/MelbournePedestrian/CNN/Asymm_1/Variable/2021-03-15T17:04:30/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Asymm_1/Variable/2021-03-15T14:46:28/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Asymm_1/Variable/2021-03-15T14:45:27/'

]

out_path_ = '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/results_self/trunc'
hyperpar = [r'$\lambda_{init}$', 'losses']
keys = ['f1_weighted']
order = ['$\\mathcal{L}_c$',
         '$\\mathcal{L}_c + \\mathcal{L}_{ae}$',
         '$\\mathcal{L}_c + \\mathcal{L}_{cc}$',
         '$\\mathcal{L}_c + \\mathcal{L}_{ae} + \\mathcal{L}_{cc}$']

init_centers = [None, 1., 10.]

for ic in init_centers:
    for noise_type, path_list in zip(['symmetric', 'asymm_1'], [path_list_symm, path_list_asymm]):

        if ic is not None:
            out_path = os.path.join(out_path_, str(ic))
        else:
            out_path = out_path_

        df_total = pd.DataFrame()
        df_losses = pd.DataFrame()
        df2 = pd.DataFrame()

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
            dataset = file.split(sep='_')[0]
            df['dataset'] = dataset

            # df = df.loc[df.init_centers == 1]
            nunique = list(df.nunique().keys()[df.nunique() == 1])
            nunique = {k: v for k, v in
                       zip(nunique, df[nunique].iloc[0].values)}

            if ic is not None:
                df = df.loc[df.init_centers == ic]

            if noise_type == 'symmetric':
                df = df.loc[df.noise < 0.7]
            else:
                df = df.loc[df.noise < 0.5]

            # if noise_type == 'symmetric':
            #    df = df.loc[df.noise <= 0.3]
            # else:
            #    df = df.loc[df.noise <= 0.2]

            # Rank per dataset
            # df['acc_rank'] = df.groupby(['noise'])['acc'].rank(method='max', ascending=True, pct=True)
            df['f1_weighted_rank'] = df.groupby(['noise', 'correct'])['f1_weighted'].rank(method='average',
                                                                                          ascending=True,
                                                                                          pct=True)

            df.loc[:, 'correct'].replace({True: 'Corrected', False: 'Noisy'}, inplace=True)
            df.rename(columns={'init_centers': r'$\lambda_{init}$',
                               'delta_end': r'$\Delta_{end}$',
                               'delta_start': r'$\Delta_{start}$',
                               'correct': 'Training Labels'}, inplace=True)

            df_total = df_total.append(df)

        print(df_total.head())

        os.makedirs(out_path, exist_ok=True)

        g = sns.catplot(data=df_total,
                        y='f1_weighted', x='noise', kind='box', hue='losses', col='Training Labels', row='dataset',
                        hue_order=order, aspect=2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('noise: {}'.format(noise_type))
        plt.savefig(os.path.join(out_path, 'f1_corrected_{}.png'.format(noise_type)))

        g = sns.catplot(data=df_total,
                        y='f1_weighted_rank', x='noise', kind='box', col='Training Labels', hue='losses',
                        hue_order=order, aspect=2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('noise: {}'.format(noise_type))
        plt.savefig(os.path.join(out_path, 'ranks_{}.png'.format(noise_type)))

        g = sns.catplot(data=df_total,
                        y='f1_weighted_rank', x='losses', kind='box', col='Training Labels', order=order, aspect=1)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('noise: {}'.format(noise_type))
        plt.savefig(os.path.join(out_path, 'ranks_total_{}.png'.format(noise_type)))

"""            df['rank_true'] = df.loc[df.correct == True].groupby(['noise'])['f1_weighted'].rank(method='average',
                                                                                          ascending=True,
                                                                                          pct=True)



            method = 'average'
            a = df.groupby(['noise', 'Training Labels', 'losses'])['f1_weighted_rank'].mean()
            df_losses = df_losses.append(a.reset_index())

            a = df.groupby(['noise', 'losses'])['rank_true'].mean()
            df2 = df2.append(a.reset_index())

        os.makedirs(out_path, exist_ok=True)

        g = sns.catplot(data=df_losses, y='f1_weighted_rank', x='noise', kind='bar', hue='losses', col='Training Labels', errwidth=0.2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('losses rank: {}'.format(nunique))
        plt.savefig(os.path.join(out_path, 'losses_rank_bar{}.png'.format(str(ic))))

        g = sns.catplot(data=df_losses, y='f1_weighted_rank', kind='bar', x='losses', hue='Training Labels', col='noise', errwidth=0.2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('losses rank: {}'.format(nunique))
        plt.savefig(os.path.join(out_path, 'losses_rank{}.png'.format(str(ic))))

        g = sns.catplot(data=df_losses.loc[df_losses['Training Labels']=='Corrected'], y='f1_weighted_rank', kind='bar', x='losses', errwidth=0.2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('losses rank: {}'.format(nunique))
        plt.savefig(os.path.join(out_path, 'losses_rank2{}.png'.format(str(ic))))


        g = sns.catplot(data=df2, y='rank_true', kind='bar', x='noise', hue='losses', errwidth=0.2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('losses rank: {}'.format(nunique))
        plt.savefig(os.path.join(out_path, 'Only_true_noise{}.png'.format(str(ic))))

        g = sns.catplot(data=df2, y='rank_true', kind='bar', x='losses', errwidth=0.2)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('losses rank: {}'.format(nunique))
        plt.savefig(os.path.join(out_path, 'Only_true{}.png'.format(str(ic))))"""
