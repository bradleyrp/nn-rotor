import torch
from torch import nn
import numpy as np

from utils import standardize


class Model_CNN_Dense(nn.Module):
    
    def __init__(self, predict_size=256, n_classes=2, **params):
        super().__init__()
        
        hidden_channels = params.get("hidden_channels", 16)
        kernel = params.get("kernel", 7)
        n_stack = params.get("n_stack", 3)
        in_ch = 3
        self.predict_size = predict_size
        
        stacks = []
        
        for i in range(n_stack):

            in_channels = in_ch if i == 0 else out_channels
            out_channels = hidden_channels  # if i == 0 else in_channels * 2

            stack = nn.Sequential(
                
                nn.Conv1d(in_channels, out_channels, kernel, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                nn.Conv1d(out_channels, out_channels, kernel, padding="same"),
                nn.BatchNorm1d(out_channels),      
                nn.ReLU(),
                
                nn.MaxPool1d(2),
            )

            stacks.append(stack)

        self.head = nn.Sequential(*stacks)

        x = torch.rand(1, in_ch, predict_size)

        N, C, L = self.head(x).shape
        assert (C, N) == (out_channels, 1)

        self.n_downsampling = 1 #  2**n_stack
        # assert L == predict_size // self.n_downsampling

        self.tail = nn.Sequential(
            nn.Linear(L * out_channels, L * out_channels),
            nn.ReLU(),
            # nn.Linear(L * out_channels, L),
            nn.Linear(L * out_channels, self.predict_size),
            nn.Sigmoid(),
            # nn.Upsample(scale_factor=self.n_downsampling)
        )
        
    def forward(self, x):
        
        h = standardize(x)
        h = self.head(h)
        h = h.flatten(1)
        y = self.tail(h)
        y = y.repeat_interleave(self.n_downsampling, dim=1)
        
        return y
    
    def predict(self, X, step):
        
        #  X: L, C
        
        size = self.predict_size
        Y = {}
        
        for i in range(0, len(X) - size + 1, step):
            x = X[i: i + size].T.values.astype(np.float32)
            x = torch.tensor(x)
            x = x.unsqueeze(0)
            with torch.no_grad():
                y = self.forward(x)
            Y[i] = y.squeeze()
            
        result = np.full((len(Y), len(X)), np.NaN)
        
        for i, (key, value) in enumerate(Y.items()):
            result[i, key: key + size] = value
            
        return result
