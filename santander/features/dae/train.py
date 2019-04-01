from fastai.vision import Learner
import torch
import torch.nn.functional as F

from .models import DenoisedAutoencoder
from .utils import ArraysItemList


def train_dae(x, device='cuda:0'):
    x_itemlist = ArraysItemList(x)
    x_itemlists = x_itemlist.split_by_rand_pct()
    labellists = x_itemlists.label_from_lists(x_itemlists.train, x_itemlists.valid)
    data = labellists.databunch(bs=32)

    device = torch.device(device)
    model = DenoisedAutoencoder().to(device)
    loss_func = F.mse_loss
    learn = Learner(data, model, loss_func=loss_func)

    learn.lr_find()
    learn.fit_one_cycle(10)

    model = learn.model.eval()
    return model
