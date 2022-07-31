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
    