from logging import getLogger

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import NNClassifier
from ...preprocess import augment
from ...utils import step_timer

logger = getLogger(__name__)


def train_nn(x_train, y_train, x_test, params, logger, n_splits=5, seed=42):
    input_size = x_train.shape[1]
    train_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    splits = list(StratifiedKFold(
        n_splits=n_splits, random_state=seed).split(x_train, y_train))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_fold_x = x_train.iloc[train_idx, :]
        train_fold_y = y_train[train_idx]
        valid_fold_x = x_train.iloc[valid_idx, :]
        valid_fold_y = y_train[valid_idx]

        with step_timer(f'fold {fold + 1}', out=logger.info):
            train_fold_x, train_fold_y = augment(train_fold_x, train_fold_y)

            model = NNClassifier(input_size, **params)
            model.fit(train_fold_x, train_fold_y, valid_fold_x, valid_fold_y)

            y_preds = model.predict_proba(valid_fold_x)[:, 1]
            train_preds[valid_idx] = y_preds
            test_preds += model.predict_proba(x_test)[:, 1] / n_splits
            logger.info(f'[fold {fold + 1}] auc score: {roc_auc_score(valid_fold_y, y_preds):<8.5f}')

    logger.info(f'CV score: {roc_auc_score(y_train, train_preds):<8.5f}')

    return train_preds, test_preds
