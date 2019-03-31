from abc import ABCMeta, abstractmethod
from pathlib import Path
import re

import pandas as pd


class Feature(metaclass=ABCMeta):
    def __init__(self, data_dir, out=print, prefix='', suffix=''):
        self.out = out
        self.prefix = prefix + '_' if prefix else ''
        self.suffix = '_' + suffix if suffix else ''
        self.name = self.convert_to_snakecase(self.__class__.__name__)
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(data_dir) / f'{self.name}_train.feature'
        self.test_path = Path(data_dir) / f'{self.name}_test.feature'

    @staticmethod
    def convert_to_snakecase(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def run(self, overwrite=False):
        if not overwrite and self.check_existence():
            self.out(f'{self.name} already exists. skipped.')
            return
        self.create_features()
        self.train.columns = self.prefix + self.train.columns + self.suffix
        self.test.columns = self.prefix + self.test.columns + self.suffix
        self.save()

    def check_existence(self):
        return self.train_path.exists() and self.test_path.exists()

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))
