import gc

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ...preprocess import augment
from ...utils import step_timer


def train_lgb(x_train, y_train, x_test, params, logger,
              n_splits=5, n_models=5, num_round=1000000, early_stop=4000, seed=2319):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)
    train_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))
    feature_importances = np.zeros(x_train.shape[1])

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
        train_fold_x = x_train.iloc[train_idx, :]
        train_fold_y = y_train[train_idx]
        valid_fold_x = x_train.iloc[valid_idx, :]
        valid_fold_y = y_train[valid_idx]

        with step_timer(f'fold {fold + 1}', out=logger.info):
            for i in range(n_models):
                with step_timer(f'model {i + 1}', out=logger.info):
                    train_fold_x_aug, train_fold_y_aug = augment(train_fold_x, train_fold_y)

                    dtrain = lgb.Dataset(train_fold_x_aug, label=train_fold_y_aug)
                    dvalid = lgb.Dataset(valid_fold_x, label=valid_fold_y, reference=dtrain)

                    clf = lgb.train(params, dtrain, num_boost_round=num_round,
                                    valid_sets=dvalid, verbose_eval=False, early_stopping_rounds=early_stop)

                    y_preds = clf.predict(x_train.iloc[valid_idx], num_iteration=clf.best_iteration)
                    train_preds[valid_idx] += y_preds / n_models
                    test_preds += clf.predict(x_test, num_iteration=clf.best_iteration) / (n_splits * n_models)

                    logger.info(f'[fold {fold + 1} model {i + 1}] best iteration: {clf.best_iteration}')
                    logger.info(f'[fold {fold + 1} model {i + 1}] auc score: {roc_auc_score(valid_fold_y, y_preds):<8.5f}')
                    feature_importances += clf.feature_importance(importance_type='gain') / (n_splits * n_models)

            logger.info(f'[fold {fold + 1}] auc score: {roc_auc_score(valid_fold_y, train_preds[valid_idx]):<8.5f}')

    logger.info(f'CV score: {roc_auc_score(y_train, train_preds):<8.5f}')
    logger.info('feature importances:')
    for i in np.argsort(feature_importances):
        logger.info(f'\t{clf.feature_name()[i]:20s} : {feature_importances[i]:>.6f}')

    return train_preds, test_preds
