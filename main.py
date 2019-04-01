from logging import getLogger
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from santander.load_data import load_data
from santander.models.lightgbm.train import train_lgb
from santander.models.nn.train import train_nn
from santander.preprocess import rank_gauss
from santander.utils import (
    Timer, step_timer, setup_logger,
    send_line_notification, send_error_to_line,
    search_weight
)

if not os.path.isdir('./output/'):
    os.mkdir('./output/')


def main_lgb():
    # set logger
    logger = getLogger(__name__)
    setup_logger(logger, './log/lgb.log')

    # set params
    n_splits = 5
    num_round = 1000000
    early_stop = 4000
    seed = 42
    features = ['main', 'dae20']

    params = {
        'bagging_freq': 5,
        'bagging_fraction': 0.331,
        'boost_from_average': False,
        'boost': 'gbdt',
        'feature_fraction': 0.0405,
        'learning_rate': 0.0063,
        'max_depth': -1,
        'metric': 'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'objective': 'binary',
        'seed': seed,
        'tree_learner': 'serial',
        'verbosity': 1
    }

    # load data
    timer = Timer(out=logger.info)
    timer.step('load data')

    train, test = load_data(features)
    x_train, y_train, train_ids = train
    x_test, test_ids = test

    # train
    timer.step('train')
    train_preds, test_preds = train_lgb(x_train, y_train, x_test, params, logger,
                                        n_splits=n_splits, num_round=num_round,
                                        early_stop=early_stop, seed=seed)

    # export to csv
    timer.step('submit')
    pd.DataFrame({
        'ID_code': train_ids,
        'target': train_preds
    }).to_csv('./output/train_preds_lgb.csv', index=False)
    pd.DataFrame({
        'ID_code': test_ids,
        'target': test_preds
    }).to_csv('./output/test_preds_lgb.csv', index=False)

    elapsed_time = timer.finish()
    message = f'''main_lgb done in {elapsed_time}.
        description: Single LGBM + augmentation + DAE features.
        cv score: {roc_auc_score(y_train, train_preds):<8.5f}'''
    send_line_notification(message)


def main_nn():
    # set logger
    logger = getLogger(__name__)
    setup_logger(logger, './log/nn.log')

    # set params
    n_splits = 5
    seed = 42
    features = ['main', 'dae20']

    params = {
        'hidden_sizes': [64, 64],
        'activation': 'swish',
        'dropout_rate': None,
        'learning_rate': 1e-3,
        'n_epochs': 30,
        'batch_size': 32,
        'device': 'cpu',
        'random_state': seed,
        'verbose': False
    }

    # load data
    timer = Timer(out=logger.info)
    timer.step('load data')

    train, test = load_data(features)
    x_train, y_train, train_ids = train
    x_test, test_ids = test

    # scale
    timer.step('scale')
    x = pd.concat([x_train, x_test], ignore_index=True, sort=False)
    x = rank_gauss(x)
    x_train = x.iloc[:len(x_train), :]
    x_test = x.iloc[len(x_train):, :]

    # train
    timer.step('train')
    train_preds, test_preds = train_nn(x_train, y_train, x_test, params, logger,
                                       n_splits=n_splits, seed=seed)

    # export to csv
    timer.step('submit')
    pd.DataFrame({
        'ID_code': train_ids,
        'target': train_preds
    }).to_csv('./output/train_preds_nn.csv', index=False)
    pd.DataFrame({
        'ID_code': test_ids,
        'target': test_preds
    }).to_csv('./output/test_preds_nn.csv', index=False)

    elapsed_time = timer.finish()
    message = f'''main_nn done in {elapsed_time}.
        description: Single NN + augmentation + DAE features.
        cv score: {roc_auc_score(y_train, train_preds):<8.5f}'''
    send_line_notification(message)


def main_ensemble():
    # set logger
    logger = getLogger(__name__)
    setup_logger(logger, './log/nn.log')

    # load data
    start_time = time.time()
    test_ids = pd.read_csv('./input/sample_submission.csv', usecols=['ID_code']).values.squeeze()
    y_train = pd.read_csv('./input/train.csv', usecols=['target'], dtype=np.int8).values.squeeze()

    train_preds_lgb = pd.read_csv('./output/train_preds_lgb.csv', usecols=['target']).values.squeeze()
    test_preds_lgb = pd.read_csv('./output/test_preds_lgb.csv', usecols=['target']).values.squeeze()
    train_preds_nn = pd.read_csv('./output/train_preds_nn.csv', usecols=['target']).values.squeeze()
    test_preds_nn = pd.read_csv('./output/test_preds_nn.csv', usecols=['target']).values.squeeze()

    # search weight
    best_weight, best_score = search_weight(train_preds_lgb, train_preds_nn, y_train)
    logger.info(f'best score: {best_score:<8.5f}')
    logger.info(f'coefficients:')
    logger.info(f'  LGBM: {best_weight:.3f}')
    logger.info(f'  NN: {1 - best_weight:.3f}')
    test_preds = best_weight * test_preds_lgb + (1 - best_weight) * test_preds_nn

    # export to csv
    pd.DataFrame({
        'ID_code': test_ids,
        'target': test_preds
    }).to_csv('./output/test_preds_ensemble.csv', index=False)

    elapsed_time = time.time() - start_time
    message = f'''main_ensemble done in {elapsed_time:.0f} sec.
        description: LGBM + NN ensemble.
        cv score: {best_score:<8.5f}'''
    send_line_notification(message)


if __name__ == '__main__':
    with send_error_to_line('function main_lgb failed.'):
        main_lgb()
