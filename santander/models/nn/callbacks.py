from pathlib import Path

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Arguments
    _________
        patience: How long to wait after last time validation loss improved.
        model_path: Where to save the model.
        out: How to output the log.
        verbose: If True, prints a message for each validation loss improvement. 
    """

    def __init__(self, patience=10, model_path='', out=print, verbose=False):
        self.patience = patience
        self.model_path = Path(model_path)
        self.out = out
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = - val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            self.out(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease.
        """
        if self.verbose:
            self.out(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path / 'checkpoint.pt')
        self.val_loss_min = val_loss
