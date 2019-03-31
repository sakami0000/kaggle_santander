import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from torch import nn
import torch.utils.data

from .layers import DenseModule
from .utils import seed_torch, sigmoid


class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 64],
                 output_size=1, activation='relu', dropout_rate=None):
        super(Net, self).__init__()
        sizes = [input_size] + list(hidden_sizes)
        modules = [
            DenseModule(sizes[i], sizes[i + 1], activation, dropout_rate)
            for i in range(len(sizes) - 1)
        ]
        modules.append(nn.Linear(sizes[-1], output_size))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        out = self.net(x)
        return out


class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_sizes=[64, 64],
                 activation='relu', dropout_rate=None,
                 learning_rate=0.001, n_epochs=5, batch_size=64,
                 device='cuda:0', random_state=None, verbose=False):
        self.model = Net(input_size=input_size,
                         hidden_sizes=hidden_sizes,
                         activation=activation,
                         dropout_rate=dropout_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.verbose = int(verbose)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.scheduler = None

        seed_torch(random_state)
        self.model.to(self.device)

    def fit(self, X, y, X_valid=None, y_valid=None):
        X = np.array(X)
        y = np.array(y)
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)

        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y[:, np.newaxis], dtype=torch.float32).to(self.device)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True)

        if self.verbose and X_valid is not None and y_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            y_valid = torch.tensor(y_valid[:, np.newaxis], dtype=torch.float32).to(self.device)
            
            valid = torch.utils.data.TensorDataset(X_valid, y_valid)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.batch_size, shuffle=False)
        elif self.verbose and (X_valid is not None or y_valid is not None):
            raise ValueError('set both X_valid and y_valid')
        else:
            valid_loader = None
            
        valid_preds = None

        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            avg_loss = 0.

            if self.scheduler:
                self.scheduler.step()

            for x_batch, y_batch in train_loader:
                y_pred = self.model(x_batch)

                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            if self.verbose and epoch % self.verbose == 0:
                if valid_loader:
                    self.model.eval()
                    valid_preds = np.zeros(X_valid.shape[0])
                    avg_val_loss = 0.
                    
                    for i, (x_batch, y_batch) in enumerate(valid_loader):
                        with torch.no_grad():
                            y_pred = self.model(x_batch).detach()
                            
                        avg_val_loss += self.loss_fn(y_pred, y_batch).item() / len(valid_loader)
                        valid_preds[i * self.batch_size:(i + 1) * self.batch_size] = sigmoid(
                            y_pred.cpu().numpy())[:, 0]
                    
                    elapsed_time = time.time() - start_time
                    print('Epoch {}/{} \t loss: {} \t valid loss: {} \t time: {:.1f}s'.format(
                        epoch + 1, self.n_epochs, avg_loss, avg_val_loss, elapsed_time))
                else:
                    elapsed_time = time.time() - start_time
                    print('Epoch {}/{} \t loss: {} \t time: {:.1f}s'.format(
                        epoch + 1, self.n_epochs, avg_loss, elapsed_time))
        
        return valid_preds

    def predict_proba(self, X):
        X = np.array(X)

        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)
        test = torch.utils.data.TensorDataset(X_test)
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        test_preds = np.zeros(len(X))

        for i, (x_batch,) in enumerate(test_loader):
            with torch.no_grad():
                y_pred = self.model(x_batch).detach()

            test_preds[i * self.batch_size:(i + 1) * self.batch_size] = sigmoid(
                y_pred.cpu().numpy())[:, 0]

        test_preds = test_preds[:, np.newaxis]
        test_preds = np.hstack((1 - test_preds, test_preds))
        return test_preds

    def predict(self, X):
        preds = self.predict_proba(X)
        return np.argmax(preds, axis=1)
