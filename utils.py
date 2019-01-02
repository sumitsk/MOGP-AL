import numpy as np
import torch
import pandas as pd

CONST = .5*np.log(2*np.pi*np.exp(1))

def to_torch(arr):
    if arr is None:
        return None
    if arr.__class__.__module__ == 'torch':
        return arr
    if arr.__class__.__module__ == 'numpy':
        return torch.FloatTensor(arr)
    return arr


def to_numpy(x):
    if x is None:
        return None
    if x.__class__.__module__ == 'torch':
        return x.detach().cpu().numpy()
    if x.__class__.__module__ == 'numpy':
        return x
    return np.array(x)


def vec_to_one_hot_matrix(vec, max_val=None):
    if max_val is None:
        max_val = np.max(vec)
    mat = np.zeros((len(vec), max_val+1))
    mat[np.arange(len(vec)), vec] = 1
    return mat


def load_data(train_fn, test_fn, input_features, target_features):
    train_df = pd.read_pickle(train_fn)
    train_x = train_df[input_features].values
    train_y = train_df[target_features].values

    test_df = pd.read_pickle(test_fn)
    test_x = test_df[input_features].values
    test_y = test_df[target_features].values
    return train_x, train_y, test_x, test_y


def zero_mean_unit_variance(data, mean=None, std=None):
    # zero mean unit variance normalization
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    return (data - mean) / std


def normalize(data, col_max=None):
    # divide each column with the corresponding max value
    col_max = data.max(0) if col_max is None else col_max
    return data/col_max


def compute_rmse(true, pred):
    return np.linalg.norm(true.squeeze() - pred.squeeze())/np.sqrt(len(true))


def compute_mae(true, pred):
    return np.mean(np.abs(true.squeeze() - pred.squeeze()))


def entropy_from_var(variance):
    return CONST + .5*np.log(variance)


def entropy_from_cov(cov):
    return cov.shape[0] * CONST + .5 * np.linalg.slogdet(cov)[1].item()