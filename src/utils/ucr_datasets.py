import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder

ucr = UCR_UEA_datasets()
# UCR/UEA univariate and multivariate datasets.


def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert (y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    if np.isnan(X_scaled).any():
        X_scaled = np.nan_to_num(X_scaled, copy=False, nan=0.0)
    return X_scaled, y


def load_data(dataset_name):
    try:
        return load_ucr(dataset_name)
    except:
        print(
            'Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets:{}.'.format(
                dataset_name, ucr.list_datasets()))
        exit(0)


def plot_class_examples():
    pass


if __name__ == '__main__':
    dataset_name = 'EthanolLevel'
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
