import pandas as pd

TRAIN_DATA = './input/train.csv'
TEST_DATA = './input/test.csv'


def load_data(features=None):
    if features:
        dfs = [pd.read_feather(f'./input/{f}_train.feather') for f in features]
        train = pd.concat(dfs, axis=1)
        dfs = [pd.read_feather(f'./input/{f}_test.feather') for f in features]
        test = pd.concat(dfs, axis=1)
    else:
        columns = [f'var_{i}' for i in range(200)]
        dtypes = dict(zip(columns, ['float32'] * 200))
        dtypes['ID_code'] = 'object'

        test = pd.read_csv(TEST_DATA, dtype=dtypes)

        dtypes['target'] = 'int32'
        train = pd.read_csv(TRAIN_DATA, dtype=dtypes)

    x_train = train.drop(['ID_code', 'target'], axis=1)
    y_train = train['target'].values
    train_ids = train['ID_code'].values

    x_test = test.drop('ID_code', axis=1)
    test_ids = test['ID_code'].values

    return (x_train, y_train, train_ids), (x_test, test_ids)
