import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBaseline(nn.Module):
    def __init__(self):
        super(ConvolutionalBaseline, self).__init__()

        # Convolution with kernel size 1 (i.e. shallow position-wise feed forward)
        # (batch_size, 2 (m/z, intensity), peaks_n) -> (batch_size, 16, peaks_n)
        self.pff = nn.Sequential(
            #nn.Conv1d(2, 64, 1),
            #nn.ReLU(),
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Convolution layer 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(64, 128, 4),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        # Convolution layer 2
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(128, 256, 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Single output feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.pff(x).transpose(-2, -1)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, 1, -1)
        #x = F.dropout(x, p=0.1)
        x = self.feed_forward(x)
        return x