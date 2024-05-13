import torch.nn as nn
from typing import Sequence
import dreams.utils.misc as utils


class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()

        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == 'interpolated':
            assert depth is not None
            hidden_dim = utils.interpolate_interval(a=in_dim, b=out_dim, n=depth - 1, only_inter=True, rounded=True)
        elif isinstance(hidden_dim, Sequence):  # e.g. is List or Tuple
            depth = len(hidden_dim)
        else:
            raise ValueError

        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)
