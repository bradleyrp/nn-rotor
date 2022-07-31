import numpy as np
from numba import njit

@njit
def convolve_square(A, n=1):
    N, M = A.shape
    assert N == M
    
    result = np.zeros(N)
    shift = n // 2
    
    for i in range(0, N - n):
        
        a = 0
        
        a += A[i, i: i + n].mean()          # top
        a += A[i + n - 1, i: i + n].mean()  # bottom
        a += A[i: i + n, i].mean()          # left          
        a += A[i: i + n, i + n - 1].mean()  # right
        
        result[i + shift] = a / 4
        
    return result


def create_proba(x, n, threshold=0.5):
    
    y = np.convolve(
        x > threshold,
        np.ones(n),
        mode="same"
    )
    
    y = np.clip(y, 0, 1)
    
    return y


def predict_1d(A, **kw):
    
    threshold = kw.get("threshold", 0.75)
    n_min = kw.get("n_min", 11)  # x10 ms
    n_step = kw.get("n_step", 2)
    
    n_max = len(A)
    proba = np.zeros(n_max)

    for n in range(n_min, n_max, n_step):

        x = convolve_square(A, n)
        proba_x = create_proba(x, n, threshold)
        proba = np.vstack([proba, proba_x]).max(axis=0)
        
    return proba
