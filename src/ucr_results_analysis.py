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
parser.add_argument('--path', type=str,
                    default='/home/castel/PycharmProjects/torchembedding/results/ucr_labelnoise_hyper_ablaton/CBF/CNN/2021-03-11/')
args = parser.parse_args()

csv = None
for file in os.listdir(args.path):
    if file.endswith('.csv'):
        print('DataFrame found: {}'.format(file))
        csv =file
        break
if csv is None:
    raise ValueError

df = pd.read_csv(os.path.join(args.path, file))

nunique = list(df.nunique().keys()[df.nunique() == 1])
nunique = {k: v for k, v in
           zip(nunique, df[nunique].iloc[0].values)}

df = drop_constant_column(df)

keys = ['acc', 'f1_weighted']
df['acc_rank'] = df.groupby(['noise'])['acc'].rank(method='dense', ascending=False)
df['f1_rank'] = df.groupby(['noise'])['f1_weighted'].rank(method='dense', ascending=False)

df['sub_f1_rank'] = df.groupby(['noise'])['f1_weighted'].rank(method='dense', ascending=False)
g = sns.catplot(x='init_centers', y='sub_f1_rank', hue='delta_start', col='delta_end', row='noise',
                        data=df, kind='bar')


