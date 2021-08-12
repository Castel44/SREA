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


my_symm_path_list = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Symm/Variable/2021-03-14T20:20:40/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Symm/Variable/2021-03-15T12:55:48/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Symm/Variable/2021-03-15T01:35:15/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Symm/Variable/2021-03-15T01:35:40/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Symm/Variable/2021-03-15T01:36:30/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Symm/Variable/2021-03-15T12:59:38/',
  '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Symm/Variable/2021-03-15T13:00:48/',
  '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/MelbournePedestrian/CNN/Symm/Variable/2021-03-15T10:17:49/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Symm/Variable/2021-03-15T01:36:59/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Symm/Variable/2021-03-15T01:32:58/'

]

sota_symm_path_list = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Plane/CNN/2021-03-14T21:02:21/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/CBF/CNN/2021-03-14T20:10:59/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Trace/CNN/2021-03-14T20:38:14/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Symbols/CNN/2021-03-14T21:28:48/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/OSULeaf/CNN/2021-03-14T21:57:16/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/FaceFour/CNN/2021-03-14T22:23:57/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/ArrowHead/CNN/2021-03-14T22:54:17/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/MelbournePedestrian/CNN/2021-03-14T23:20:29/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Epilepsy/CNN/2021-03-15T00:18:34/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/NATOPS/CNN/2021-03-15T00:42:01/'

]

my_asymm_path_list = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Asymm_1/Variable/2021-03-14T21:28:27/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/Asymm_1/Variable/2021-03-15T15:54:11/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Trace/CNN/Asymm_1/Variable/2021-03-15T02:58:43/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Symbols/CNN/Asymm_1/Variable/2021-03-15T04:22:57/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/OSULeaf/CNN/Asymm_1/Variable/2021-03-15T04:37:06/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/FaceFour/CNN/Asymm_1/Variable/2021-03-15T14:46:28/',
  '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/ArrowHead/CNN/Asymm_1/Variable/2021-03-15T14:45:27/',
  '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/MelbournePedestrian/CNN/Asymm_1/Variable/2021-03-15T17:04:30/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Epilepsy/CNN/Asymm_1/Variable/2021-03-15T03:42:53/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/NATOPS/CNN/Asymm_1/Variable/2021-03-15T03:51:02/'

]

sota_asymm_path_list = [
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Plane/CNN/2021-03-15T02:08:20/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/CBF/CNN/2021-03-15T01:18:19/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Trace/CNN/2021-03-15T01:45:31/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Symbols/CNN/2021-03-15T10:55:23/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/OSULeaf/CNN/2021-03-15T11:39:41/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/FaceFour/CNN/2021-03-15T12:13:00/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/ArrowHead/CNN/2021-03-15T12:50:07/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/MelbournePedestrian/CNN/2021-03-15T13:21:45/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Epilepsy/CNN/2021-03-15T14:36:05/',
    '/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/NATOPS/CNN/2021-03-15T15:05:47/'

]

out_path_ = '/home/castel/PycharmProjects/torchembedding/results/ucr_results/main/trunc'
my_method = r'$\mathcal{L}_c + \mathcal{L}_{ae} + \mathcal{L}_{cc}$'
keys = ['f1_weighted']
order = ['CrossEntropy', 'Mixup', 'MixUp-BMM', 'Co-teaching', 'SIGUA', my_method]
ic_ = [None, 1., 10.]

for ic in ic_:
    for noise_type, path_lists in zip(['symmetric', 'asymm_1'], [[my_symm_path_list, sota_symm_path_list],
                                                                 [my_asymm_path_list, sota_asymm_path_list]]):
        if ic is not None:
            out_path = os.path.join(out_path_, str(ic))
        else:
            out_path = out_path_

        df_main = pd.DataFrame()

        for my_path, sota_path in zip(*path_lists):
            csv = None
            for file in os.listdir(sota_path):
                if file.endswith('.csv'):
                    print('DataFrame found: {}'.format(file))
                    csv = file
                    break
            if csv is None:
                raise ValueError

            df_sota = pd.read_csv(os.path.join(sota_path, file))
            df_my = pd.read_csv(os.path.join(my_path, file))
            dataset = file.split(sep='_')[0]

            if noise_type == 'symmetric':
                df_my = df_my.loc[df_my.noise < 0.7]
                df_sota = df_sota.loc[df_sota.noise < 0.7]
            else:
                df_my = df_my.loc[df_my.noise < 0.5]
                df_sota = df_sota.loc[df_sota.noise < 0.5]


            order = ['CrossEntropy', 'Mixup', 'MixUp-BMM', 'Co-teaching', 'SIGUA', my_method]

            df_sota.rename(columns={'correct': 'Algorithm'}, inplace=True)
            df_sota.loc[:, 'Algorithm'].replace({'none': 'CrossEntropy'}, inplace=True)
            df_sota.drop('losses', axis=1, inplace=True)

            df_tmp = df_my.loc[df_my.correct == True].loc[df_my.losses == my_method]
            df_tmp['Algorithm'] = my_method
            df_ = pd.concat([df_sota, df_tmp], join='inner')
            df_['dataset'] = dataset

            df_['rank'] = df_.groupby(['noise'])['f1_weighted'].rank(method='average', ascending=True, pct=True)

            a = df_.groupby(['noise', 'Algorithm'])['rank'].mean()
            df_main = df_main.append(a.reset_index())

        os.makedirs(out_path, exist_ok=True)

        g = sns.catplot(data=df_main, y='rank', x='noise', hue='Algorithm', kind='box', hue_order=order, height=6,
                        aspect=1.8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Noise:{}'.format(noise_type))
        #plt.savefig(os.path.join(out_path, 'alg_rank_noise_{}.png'.format(noise_type)))

        g = sns.catplot(data=df_main, y='rank', x='Algorithm', kind='box', order=order, height=6, aspect=1.25)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Noise:{}'.format(noise_type))
        #plt.savefig(os.path.join(out_path, 'alg_rank_{}.png'.format(noise_type)))
