from fastai.callbacks.hooks import hook_output
import numpy as np
import pandas as pd
from tqdm import trange

from .base import Feature
from .dae.train import train_dae
from .dae.utils import ArraysItemList
from ..load_data import load_data
from ..preprocess import rank_gauss


class Main(Feature):

    def create_features(self):
        self.out(f'[{self.name}] loading data')
        train, test = load_data()
        x_train, y_train, train_ids = train
        x_test, test_ids = test

        x_train['target'] = y_train
        x_train['ID_code'] = train_ids
        x_test['ID_code'] = test_ids

        self.train = x_train
        self.test = x_test


class DAE(Feature):

    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def create_features(self):
        self.out(f'[{self.name}] loading data')
        train, test = load_data()
        x_train, _, _ = train
        x_test, _ = test

        x = pd.concat([x_train, x_test], ignore_index=True, sort=False)
        x = rank_gauss(x)

        self.out(f'[{self.name}] training')
        self.model = train_dae(x, self.device)

        self.out(f'[{self.name}] extracting features')
        x_train = x.iloc[:len(x_train), :]
        x_test = x.iloc[len(x_train):, :]

        self.train = self.extract_features(x_train)
        self.train = pd.DataFrame(self.train)

        self.test = self.extract_features(x_test)
        self.test = pd.DataFrame(self.test)

    def hooked_backward(self, xb):
        with hook_output(self.model.encoder[0]) as hook_a: 
            with hook_output(self.model.encoder[1]) as hook_b: 
                with hook_output(self.model.encoder[2]) as hook_c: 
                    with hook_output(self.model.decoder[0]) as hook_d: 
                        with hook_output(self.model.decoder[1]) as hook_e: 
                            _ = self.model(xb)
        return hook_a, hook_b, hook_c, hook_d, hook_e

    def extract_features(self, x):
        x_itemlist = ArraysItemList(x)
        x_itemlists = x_itemlist.split_none()
        labellists = x_itemlists.label_from_lists(x_itemlists.train, [])
        data = labellists.databunch(bs=32)

        len_data = len(data.train_ds)
        result = np.empty((len_data, 50))

        for i in trange(len_data):
            x, _ = data.train_ds[i]
            xb, _ = data.one_item(x)
            xb = xb.to(self.device)
            _, _, hook_c, _, _ = self.hooked_backward(xb)
            result[i] = hook_c.stored[0].cpu()

        return result
