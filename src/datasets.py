import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.metrics import euclidean_distances


class Dataset2D(Dataset):
    
    def __init__(
        self,
        df,
        window_size=256,
        window_step=128,
        train=False
    ):
        
        self.df = df.copy()
        self.window_size = window_size
        self.window_step = window_step
        self.train = train
        
        self.indices = list(range(
            0,
            len(self.df) + 1 - self.window_size,
            self.window_step
        ))
        
    def create_sample(self, df, α=1.1):

        n = len(df)
        xyz = ["x", "y", "z"]
                
        image = euclidean_distances(df[xyz])
        image /= 100
        
        counts = df.period.value_counts()
        counts = (counts * 10 / counts.index > α) & (counts.index > 0)
        
        mask = np.zeros((n, n))
        for p in df.period.unique():
            if p == 0:
                continue
            if not counts[p]:
                continue
            m = (df.period == p).astype(float)
            m = np.tile(m, (len(m), 1))
            m = m * m.T
            mask = mask + m
            
        return image, mask
                        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        i1 = self.indices[idx]
        i2 = i1 + self.window_size
        
        df_slice = self.df[i1: i2]
        
        if self.train and (np.random.rand() > 0.5):
            df_slice = df_slice[::-1]
        
        x, y = self.create_sample(df_slice)
        
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)
        
        return x, y
