import torch
import torch.nn as nn
import statistics as stats
import numpy as np
import pandas as pd


class RandomDiscreteBaseline(nn.Module):
    # TODO: multivariate (e.g. for chemical formula)
    def __init__(self, y_train):
        super(RandomDiscreteBaseline, self).__init__()
        y_train = pd.Series(y_train)
        value_counts = y_train.astype('int').value_counts(normalize=True)
        self.probs = value_counts.tolist()
        self.vals = value_counts.index.tolist()

    def forward(self, x):
        batch_size = x.shape[0]
        x = np.random.choice(self.vals, batch_size, replace=True, p=self.probs)
        return torch.FloatTensor(x).unsqueeze(-1)