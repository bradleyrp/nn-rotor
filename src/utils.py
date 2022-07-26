from torch import nn
import numpy as np


def proba2class(proba):
    return (proba > 0.5).int()


def standardize(X, ϵ=1e-9):
    # X : [BATCH x CHANNEL x LENGTH]
    X_mean = X.mean(dim=2, keepdim=True)
    X_std = X.std(dim=2, keepdim=True)
    return (X - X_mean) / (X_std + ϵ)


def label_smoothing(y, α=0.):
    return y * (1 - α) + α / 2


def find_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "σ":
        return nn.Sigmoid()
    else:
        None
    

def create_diag(size, n=0): 
    x = np.ones(size)
    A = np.zeros((size, size))
    for i in range(-n, n+1):
        if i == 0:
            a = np.diag(x, i)
        elif i < 0:
            a = np.diag(x[:i], i)
        else:
            a = np.diag(x[i:], i)
        A = A + a
        
    A = A / A.sum(axis=0, keepdims=True)
    
    return A


def predict_1d(pred_2d):
    M = create_diag(len(pred_2d), 4)
    pred = (pred_2d * M).sum(axis=0)
    return pred
