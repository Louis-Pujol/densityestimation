import numpy as np

def local_rodeo(x, X, h0=1.0, beta=.9, a=1.0):
    
    N, d = X.shape
    C = 4 * np.sqrt(1 / 2 * np.sqrt(np.pi)) * np.power( 5 / (8 * np.sqrt(np.pi)), (d-1)/2 )
    actives = np.ones(d, dtype=int)
    bandwidths = h0 * np.ones(d)
    
    
    while(np.sum(actives) != 0 and np.prod(bandwidths) >= np.log(N)/N):
        
        for j in np.where(actives==1)[0]:
            
            diff = X-x/bandwidths
            
            Zhj = -1.0 * np.mean( ((diff[:,j] ** 2 / bandwidths[j]) - 1) * np.exp(np.sum(diff**2 / 2 , axis=1)) / np.power(2*np.pi, d/2)  )
            threshold = C * np.sqrt( np.power(np.log(N), a) / (N * bandwidths[j]**2 * np.prod(bandwidths) ) )
                                    
            if (np.abs(Zhj) > threshold):
                
                bandwidths[j] *= beta
            else:
                actives[j] = 0
                
    return bandwidths
            
            
    
    
    
