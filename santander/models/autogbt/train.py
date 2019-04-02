from autogbt import AutoGBTClassifier
import pandas as pd

from .objective import CustomObjective


def train_autogbt(x_train, y_train, x_test,
                  n_trials=1000, seed=2319):
    y_train = pd.Series(y_train)
    clf = AutoGBTClassifier(n_trials=n_trials,
                            objective=CustomObjective(),
                            random_state=seed)
    clf.fit(x_train, y_train)
    train_preds = clf.predict_proba(x_train)
    test_preds = clf.predict_proba(x_test)
    return train_preds, test_preds
