import torch
import torch.nn as nn
import statistics as stats
import numpy as np


class RandomContinuousBaseline(nn.Module):
    # TODO: multivariate
    # TODO: binning intsead of normal distribution?
    def __init__(self, y_train):
        super(RandomContinuousBaseline, self).__init__()
        self.mu = stats.mean(y_train)
        self.sigma = stats.stdev(y_train) 

    def forward(self, x):
        batch_size = x.shape[0]
        x = np.random.normal(self.mu, self.sigma, batch_size)
        return torch.FloatTensor(x).unsqueeze(-1)