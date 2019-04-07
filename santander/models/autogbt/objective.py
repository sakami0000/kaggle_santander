from autogbt import Objective


class CustomObjective(Objective):

    def get_param(self, trial):
        return {
            'learning_rate': 0.01,
            'num_threads': 8,
            'num_leaves': trial.suggest_int('num_leaves', 10, 80),
            'feature_fraction': trial.suggest_uniform(
                'feature_fraction', 0.01, 0.2),
            'bagging_fraction': trial.suggest_uniform(
                'bagging_fraction', 0.2, 0.5),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'colsample_bytree': trial.suggest_discrete_uniform(
                'colsample_bytree', 0.5, 1.0, 0.1),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 5.0),
            'min_data_in_leaf': trial.suggest_int('min_child_weight', 5, 100),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
            'min_sum_hessian_in_leaf': trial.suggest_uniform(
                'min_sum_hessian_in_leaf', 1, 100)
        }
