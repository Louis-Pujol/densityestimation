import numpy as np
import matplotlib.pyplot as plt
import json

import kde_cpu as kde_cpu
import kde_gpu as kde_gpu

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
        

class Graphical_KDE:
    
    def __init__(self, h):
        self.a = 1
        


        
def complete_graph_weigths(X_train, h1, h2, grid_size=20, gpu=False, n_jobs=1, held_out=False, X_test=None):
    
    _, d = X_train.shape
    
    if gpu:
        method = "gpu"
    else:
        method = "cpu"

    partition = [ [i] for i in range(d) ]
    partition += [ [j, i] for i in range(d) for j in range(i)  ]
    
    grid_1d = np.linspace(0, 1, grid_size)
    grid_2d = np.array([[i, j] for i in grid_1d for j in grid_1d])

    pn1 = {} ##
    if held_out:
        pn2 = {}
        
    for p in partition:
        loc_d = len(p)
        if loc_d == 1:
            loc_grid = grid_1d.reshape(grid_size, 1)
            h = h1
        else:
            loc_grid = grid_2d
            h = np.array([h1, h2])
        
        pn1[str(p)] = KernelDensity(X=X_train[:, p], bw=h).score(x=loc_grid, method=method, n_jobs=n_jobs, log=False)
        if held_out:
            pn2[str(p)] = KernelDensity(X=X_test[:, p], bw=h).score(x=loc_grid, method=method, n_jobs=n_jobs, log=False)
            
    
    weights = {} # \hat{I}
    for a in [ [j, i] for i in range(d) for j in range(i)  ]:
        
        
        if not held_out :
            
            pn1xy = pn1[str(a)]
            pn1x = np.repeat(pn1[str([a[0]])], grid_size)
            pn1y = np.tile(pn1[str([a[1]])], grid_size)
            
            weights[str(a)] = np.mean(pn1xy * np.log(pn1xy / (pn1x * pn1y)))
        
        else:
            
            pn1xy = pn1[str(a)]
            pn2xy = pn1[str(a)]
            pn1x = np.repeat(pn1[str([a[0]])], grid_size)
            pn1y = np.tile(pn1[str([a[1]])], grid_size)
            
            weights[str(a)] = np.mean(pn2xy * np.log(pn1xy / (pn1x * pn1y)))
            
            
        
    return weights


def kruskal(weights, threshold = None):
    
    #Find the dimension (maximum value in weigths.keys() + 1)
    d = 0
    for i in weights.keys():
        d = np.max([d] + json.loads(i))
    d += 1
    
    cliques = [ [i] for i in range(d)] #At the beginning : every variable alone
    graph = [] # Will be the final graph (list of edges)
    graph_weights = []  # corresponding weights ( \hat{I} )
    
    sort_orders = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    if threshold != None :
        sort_orders2 = []
        for i in sort_orders:
            if i[1] > threshold:
                sort_orders2.append(i)
        sort_orders = sort_orders2
    
    
    for (pairs, s) in sort_orders:
        
        a, b = json.loads(pairs)[0], json.loads(pairs)[1]
        
                
        combine = True
        clique_a = []
        clique_b = []
            
        for c in cliques:
            if a in c and b in c: #Si l'arete ajoute un cycle
                combine = False
            if a in c:
                clique_a = c
            if b in c:
                clique_b = c
                
        if combine == True: 
            graph.append([a, b])
            graph_weights.append(s)
            #MàJ de cliques
            cliques = [c for c in cliques if c != clique_a and c != clique_b]
            cliques.append(clique_a + clique_b)
                        
                
    return(graph, graph_weights)

def partitions(weights, max_size, threshold = None):
    
    #Find the dimension (maximum value in weigths.keys() + 1)
    d = 0
    for i in weights.keys():
        d = np.max([d] + json.loads(i))
    d += 1
    
    cliques = [ [i] for i in range(d)] #At the beginning : every variable alone
    graph = [] # Will be the final graph (list of edges)
    graph_weights = []  # corresponding weights ( \hat{I} )
    
    sort_orders = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    if threshold != None :
        sort_orders2 = []
        for i in sort_orders:
            if i[1] > 0.1:
                sort_orders2.append(i)
        sort_orders = sort_orders2
    
    for (pairs, s) in sort_orders:
        
        a, b = json.loads(pairs)[0], json.loads(pairs)[1]
        
                
        combine = True
        clique_a = []
        clique_b = []
            
        for c in cliques:
            if a in c:
                clique_a = c
                if len(c) >= max_size: #if a is not alone
                    combine = False
            if b in c:
                clique_b = c
                if len(c) >= max_size: #if a is not alone
                    combine = False
                    
        if clique_a == clique_b:
            combine = False
        if len(clique_a) + len(clique_b) > max_size:
            combine = False
                
        if combine == True: 
            graph.append([a, b])
            graph_weights.append(s)
            #MàJ de cliques
            cliques = [c for c in cliques if c != clique_a and c != clique_b]
            cliques.append(clique_a + clique_b)
                        
                
    return cliques


def show_graph(graph, axes):
    
    d = len(axes)
    
    for i in range(d):
    
        plt.text( np.cos(i*np.pi*2/d)+0.05, np.sin(i*np.pi*2/d)+0.01, "(" + str(i) + ")" + axes[i] )

        for j in range(i):

            if [j, i] in graph:

                plt.plot( [np.cos(i*np.pi*2/d), np.cos(j*np.pi*2/d)],
                         [np.sin(i*np.pi*2/d), np.sin(j*np.pi*2/d)],
                         color='k')

    plt.scatter( np.cos(np.array(range(d))*np.pi*2/d), 
                np.sin(np.array(range(d))*np.pi*2/d), s=200, c="k" )
    plt.show()

    


def logdensity_from_tree(X, graph, h):
    N, d = X.shape
    log_dens = np.zeros(N)
    def degree(graph, i):
        return( len([ e for e in graph if i in e ]) )
    
    for k in range(d):
        if degree(graph, [k]) != 1:
            log_dens += (1 - degree(graph, [k])) * KernelDensity(X=X[:, [k]],
                                                                   bw=h*np.ones(1)).score(x=X[:, [k]],
                                                                                              method='gpu', log=True)
    for e in graph:
        log_dens += KernelDensity(X=X[:, e], bw=h*np.ones(2)).score(x=X[:, e], method='gpu', log=True)


    return log_dens

def logdensity_from_partition(X, partition, h):
    N, d = X.shape
    log_dens = np.zeros(N)
    
    for p in partition:
        log_dens += KernelDensity(X=X[:, p], bw=h*np.ones(len(p))).score(x=X[:, p], method='gpu', log=True)

    return log_dens
