import numpy as np
from sklearn.preprocessing import MinMaxScaler


#Class Regular_histogram
class Regular_Histogram():
    
    '''
    args :    - structure : list of list, partition of {0, 1, ..., d-1} corresponding to independence structure
              - n_bins : number of bins in every dimensional
           
    methods : - fit : compute histogram and empirical log-likelihood
              - test_log_likelihood : compute log_likelihood on another set (must be call after fit), drawback : +1 in each bin to avoid log(0)...
    
    /!\ all data will be normalized to fill [0, 1]^d
    TODO : add quadratic_loss, possibility set different n_bins to different set of structure
    '''
    
    def __init__(self, structure, n_bins):
        self.structure = structure
        self.n_bins = n_bins
        self.h = None
        
        #TODO : error if invalid structure's shape
        if isinstance(structure[0], int):
            self.d = len(structure)
        else:
            self.d = 0
            for p in self.structure:
                self.d += len(p)
        
    def fit(self, X):
        '''compute histogram and empirical log-likelihood
        '''
        
        X = MinMaxScaler().fit_transform(X) # Renormalization of X
        n, dim = X.shape
        self.n = n
        if dim != self.d:
            print("error, data must be " + str(self.d) + "-dimensional")
            return 0
        
        if isinstance(self.structure[0], int):
            self.h, _ = np.histogramdd(X, bins=self.n_bins)
            self.log_likelihood = compute_log_likelihood(self.h)
        else:
            self.h = []
            self.log_likelihood = 0
            for j in self.structure:
                loc_h, _ = np.histogramdd(X[:, j], bins=self.n_bins)
                self.h.append(loc_h)
                self.log_likelihood += compute_log_likelihood(loc_h)
            
    def test_log_likelihood(self, X_test, opt=1, lambd=1):
        ''' Compute test error on X_test
            see overleaf to option to avoid +infty
        '''
        X_test = MinMaxScaler().fit_transform(X_test) # Renormalization of X_test
        n_test, dim = X_test.shape
        if dim != self.d:
            print("error, data must be " + str(self.d) + "-dimensional")
            return 0

        #Compute h_test
        if isinstance(self.structure[0], int):
            h_test, _ = np.histogramdd(X_test, bins=self.n_bins)
        else:
                h_test = []
                for j in self.structure:
                    loc_h, _ = np.histogramdd(X_test[:, j], bins=self.n_bins)
                    h_test.append(loc_h)

        ll_test = 0

        if isinstance(self.structure[0], int):
            
            if opt == 1:
                for i in cartesian(self.d * (np.array(range(self.n_bins)),)):
                    ll_test += h_test[tuple(i)] * np.log( (self.h[tuple(i)] + lambd) * (self.n_bins ** self.d) / self.n) / n_test
            elif opt == 2:
                for i in cartesian(self.d * (np.array(range(self.n_bins)),)):
                    if h_test[tuple(i)] != 0 and self.h[tuple(i)] != 0:
                        ll_test += h_test[tuple(i)] * np.log(self.h[tuple(i)] * (self.n_bins ** self.d) / self.n) / n_test
                    elif h_test[tuple(i)] != 0 and self.h[tuple(i)] == 0: #Theoritically infty
                                
                        ll_test -= lambd * h_test[tuple(i)]

        else:            
            for j in range(len(self.structure)):
                d_loc = len(self.structure[j])
                if opt == 1:
                    for i in cartesian(d_loc * (np.array(range(self.n_bins)),)):
                        ll_test += h_test[j][tuple(i)] * np.log( (self.h[j][tuple(i)] + lambd) * (self.n_bins ** d_loc) / self.n ) / n_test
                
                if opt == 2:
                    for i in cartesian(d_loc * (np.array(range(self.n_bins)),)):
                        if h_test[j][tuple(i)] != 0 and self.h[j][tuple(i)] != 0:
                            ll_test += h_test[j][tuple(i)] * np.log(self.h[j][tuple(i)] * (self.n_bins ** d_loc) / self.n ) / n_test
                        elif h_test[j][tuple(i)] != 0 and self.h[j][tuple(i)] == 0: #Theoritically infty
                            
                            ll_test -= lambd * h_test[j][tuple(i)]
                        

        return ll_test
    
    
    def complete_histogram(self):
        '''For computation with L2 norm it is useful to have access to the full histogram
           and not only piecewise on each set of the variables partition
           #TODO: tbw
        '''
        
        full_h = np.zeros( self.d * (self.n_bins, ) )
        print(full_h.shape)
        
        
    
            
#Ancilliaries function
def cartesian(arrays):
    
    ''' return a list of all combinaison between vectors of arrays'''
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix



def compute_log_likelihood(h):
    d = len((h.shape))
    
    if d==1:
        bins = len(h)
    else:
        bins = len(h[0])
    n = np.sum(h)
    h_raw = h.reshape(np.prod(h.shape))
    h_raw = h_raw[h_raw != 0]
    
    ll = np.mean(np.log( ((bins) ** d) * h_raw / n))
    return ll
