import numpy as np
import torch
from pykeops.numpy import LazyTensor as LazyTensor_np
import time


def eval_gpu(x, X, h):
    
    N, d = X.shape
    x_i = LazyTensor_np(x[:, None, :])  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
    X_j = LazyTensor_np(X[None, :, :])  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y
    h_l = LazyTensor_np(h)

    D_ij = ( -0.5 * (((x_i - X_j) / h_l) ** 2).sum(-1))  # **Symbolic** (M, N) matrix of squared distances
    s_i = D_ij.exp().sum(dim=1).ravel()  # genuine (M,) array of integer indices
    
    out = s_i / (N*np.prod(h)*np.power(2*np.pi, d/2))
    
    return out

def eval_gpu_matrix(x, X, h):
    
    N, d = X.shape
    h_inv = np.linalg.inv(h)
    x_i = LazyTensor_np(x[:, None, :])  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
    X_j = LazyTensor_np(X[None, :, :])  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y
    h_l = LazyTensor_np(h_inv, axis=2) 

    D_ij = ( -0.5 * (( h_l.tensorprod(x_i - X_j )) ** 2).sum(-1))  # **Symbolic** (M, N) matrix of squared distances
    s_i = D_ij.exp().sum(dim=1).ravel()  # genuine (M,) array of integer indices
    
    out = s_i / (N*np.linalg.det(h)*np.power(2*np.pi, d/2))
    
    return out
    


if __name__ == '__main__':
    
    
    n, N = 100,100
    d = 5
    
    x = np.random.rand(n, d)
    X = np.random.rand(N, d)
    h = 0.2*np.ones(d)
    
    start = time.time()
    out = eval_gpu(x, X, h)
    print(out[0:10])
    print(time.time() - start)
    
    start = time.time()
    out = eval_gpu_matrix(x, X, 0.2*np.identity(d))
    print(out[0:10])
    print(time.time() - start)
    
    
