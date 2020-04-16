import numpy as np
import includes.kde_cpu as kde_cpu
import includes.kde_gpu as kde_gpu


class KernelDensity:
    
    def __init__(self, X, bw = None):
        
        self.X = X
        self.N, self.d = X.shape
            
        self.bw = bw
        
    def score(self, x, method='cpu', n_jobs=1, log=False):
        
        if method == 'cpu' and not log:
            return np.array(kde_cpu.evaluate_bwvector(x, self.X, self.bw, n_jobs))
        elif method == 'cpu' and log:
            return( np.log(kde_cpu.evaluate_bwvector(x, self.X, self.bw, n_jobs)) )
        elif method == 'gpu' and not log:
            return kde_gpu.eval_gpu(x, self.X, self.bw)
        elif method == 'gpu' and log:
            return np.log(kde_gpu.eval_gpu(x, self.X, self.bw))
            
        
if __name__=="__main__":
    
    from sklearn.neighbors import KernelDensity as skKernelDensity
    from time import time
    
    N, d = (10000, 3)
    h = 0.2
    X = np.random.rand(N, d)
    
    start = time()
    test1 = KernelDensity(X=X, bw=h*np.ones(d)).score(x=X, method='cpu', n_jobs=-1, log=True)
    print(time()-start)
    print(test1[0:10])
    
    start = time()
    test2 = KernelDensity(X=X, bw=h*np.ones(d)).score(x=X, method='gpu', n_jobs=-1, log=True)
    print(time()-start)
    print(test2[0:10])
    
    start = time()
    test3 = skKernelDensity(bandwidth=h, kernel="gaussian").fit(X).score_samples(X)
    print(time()-start)
    print(test3[0:10])
    
