import numpy as np
from sklearn.model_selection import StratifiedKFold

from .models import NNClassifier


def train_nn(x_train, y_train, x_test, n_splits=5, seed=42):
    input_size = x_train.shape[1]
    train_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    splits = list(StratifiedKFold(
        n_splits=n_splits, random_state=seed).split(x_train, y_train))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {fold + 1}')

        train_fold_x = x_train[train_idx, :]
        train_fold_y = y_train[train_idx]
        valid_fold_x = x_train[valid_idx, :]
        valid_fold_y = y_train[valid_idx]

        model = NNClassifier(input_size)
        model.fit(train_fold_x, train_fold_y, valid_fold_x, valid_fold_y)

        train_preds[valid_idx] = model.predict_proba(valid_fold_x)[:, 1]
        test_preds += model.predict_proba(x_test)[:, 1] / n_splits

    return train_preds, test_preds
