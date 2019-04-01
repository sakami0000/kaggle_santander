import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import rankdata


def rank_gauss(x, epsilon=0.001):
    lower = -1 + epsilon
    upper = 1 - epsilon
    scale_range = upper - lower

    i = np.argsort(x, axis=0)
    j = np.argsort(i, axis=0)

    assert (j.min() == 0).all()
    assert (j.max() == len(j) - 1).all()

    j_range = len(j) - 1
    divider = j_range / scale_range

    transformed = j / divider
    transformed = transformed - upper
    transformed = erfinv(transformed)

    return transformed


def augment(x, y, t=2):
    df = False
    if isinstance(x, pd.DataFrame):
        df = True
        columns = x.columns
        x = x.values

    x = np.array(x)
    y = np.array(y)

    xs, xn = [], []
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xs.append(x1)

    for i in range(t // 2):
        mask = y == 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xs, xn])
    y = np.concatenate([y, ys, yn])

    if df:
        x = pd.DataFrame(x, columns=columns)
    return x, y


def rank_scale(pred):
    return rankdata(pred) / len(pred)


def add_features(x_train, x_test):
    x_train['diff_21_22'] = x_train['var_22'] - x_train['var_21']
    x_test['diff_21_22'] = x_test['var_22'] - x_test['var_21']
    return x_train, x_test
