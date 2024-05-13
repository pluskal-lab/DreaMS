import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBaseline(nn.Module):
    # TODO: arbitrary depth

    def __init__(self, in_size, hidden_size, out_size):
        super(FeedForwardBaseline, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        return self.feed_forward(x)


# class FeedForwardBaseline(nn.Module):
#     def __init__(self, in_size, hidden_size, n):
#         super(FeedForwardBaseline, self).__init__()
#         self.ff = nn.Sequential(
#             nn.Linear(in_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#         )
#         self.ff_out = nn.Linear(n, 1)
#
#     def forward(self, x):
#         x = self.ff(x)
#         x = self.ff_out(x.squeeze(-1))
#         return x.squeeze(-1)
