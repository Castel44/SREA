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


parser = argparse.ArgumentParser()
parser.add_argument('--sota_path', type=str,
                    default='/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_SOTA/Plane/CNN/2021-03-14T21:02:21/'
                    )
parser.add_argument('--my_path', type=str,
                    default='/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/Plane/CNN/Symm/Variable/2021-03-14T20:20:40/'
                    )
args = parser.parse_args()

csv = None
for file in os.listdir(args.sota_path):
    if file.endswith('.csv'):
        print('DataFrame found: {}'.format(file))
        csv = file
        break
if csv is None:
    raise ValueError

df_sota = pd.read_csv(os.path.join(args.sota_path, file))
df_my = pd.read_csv(os.path.join(args.my_path, file))

dataset = file.split(sep='_')[0]
noise_type = df_my.noise_type[0]
outpath = '/home/castel/PycharmProjects/torchembedding/results/'
my_method = r'$\mathcal{L}_c + \mathcal{L}_{ae} + \mathcal{L}_{cc}$'

keys = ['f1_weighted']
order = ['CrossEntropy', 'Mixup', 'MixUp-BMM', 'Co-teaching', 'SIGUA', my_method]
outdir_ = os.path.join(outpath, 'ucr_results', 'trunc', dataset, noise_type)

df_sota.rename(columns={'correct': 'Algorithm'}, inplace=True)
df_sota.loc[:, 'Algorithm'].replace({'none': 'CrossEntropy'}, inplace=True)
df_sota.drop('losses', axis=1, inplace=True)

init_centers = [None, 1., 10.]

if noise_type == 'Symm':
    df_my = df_my.loc[df_my.noise < 0.7]
    df_sota = df_sota.loc[df_sota.noise < 0.7]
else:
    df_my = df_my.loc[df_my.noise < 0.5]
    df_sota = df_sota.loc[df_sota.noise < 0.5]

for ic in init_centers:

    df_tmp = df_my.loc[df_my.correct == True].loc[df_my.losses == my_method]

    if ic is not None:
        df_tmp = df_tmp.loc[df_tmp.init_centers == ic]

    df_tmp['Algorithm'] = my_method
    df = pd.concat([df_sota, df_tmp], join='inner')

    for k in keys:

        if ic is not None:
            outdir = os.path.join(outdir_, str(ic))
        else:
            outdir = outdir_

        dirpath = os.path.join(outdir, k)
        os.makedirs(dirpath, exist_ok=True)
        s = 'Dataset: {} - Noise type: {}'.format(dataset, noise_type)
        g = sns.catplot(data=df, x='noise', y=k, hue='Algorithm', hue_order=order, kind='box', height=8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(s)
        plt.savefig(os.path.join(dirpath, '{}_{}.png'.format(k, str(ic))))
        # plt.show(block=True)

        df['{}_rank'.format(k)] = df.groupby(['noise'])[k].rank(method='average', ascending=True, pct=True)

        g = sns.catplot(data=df, y='{}_rank'.format(k), x='noise', hue='Algorithm', kind='bar', errwidth=0.2,
                        hue_order=order, height=8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Dataset: {} - Noise type: {} - methods rank: {}'.format(dataset, noise_type, k))
        plt.savefig(os.path.join(dirpath, 'methods_{}_rank_noise_{}.png'.format(k, str(ic))))

        g = sns.catplot(data=df, y='{}_rank'.format(k), x='Algorithm', kind='bar', errwidth=0.2, order=order, height=8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Dataset: {} - Noise type: {} - methods rank: {}'.format(dataset, noise_type, k))
        plt.savefig(os.path.join(dirpath, 'methods_{}_rank_{}.png'.format(k, str(ic))))
