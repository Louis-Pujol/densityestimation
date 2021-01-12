import numpy as np
import gudhi.point_cloud.knn as knn
import gudhi.point_cloud.dtm as dtm
import gudhi.clustering.tomato as tm
import itertools
import more_itertools as mit
import ast

from sklearn.model_selection import train_test_split

def compute_hat_Hs_dtm(X, train_size, test_size, max_size, k, q):
    #Version simple où k est fixé à l'avance
    
    N, d = X.shape
    
    scores = {}
    X_train, X_test = train_test_split(X, train_size=train_size, test_size=test_size)
    
    sets = []
    for i in range(1, max_size+1):
        sets += list(itertools.combinations(range(d), i))
    
    for S in sets:
        
        scores[S] = np.mean(np.log(dtm.DTMDensity(k=k, weights=None,
                                                q=q, dim=None,
                                                normalize=True).fit(X_train[:, S]).transform(X_test[:, S])))
            
    return scores

def compute_score_by_partition_dtm(X, max_size, Hs):
    #Version simple où k est fixé à l'avance
    
    N, d = X.shape
    lst = list(range(d))
    # Output : best partition for each max_size
    output = {}
    
    partitions = [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]
    keep = np.where(np.array([np.max([len(i) for i in p]) for p in partitions]) <=  max_size )[0]
    partitions_keep = [partitions[k] for k in keep]
    del partitions
    
    for partition in partitions_keep:
        
        output[str(partition)] = 0
        for p in partition:
            output[str(partition)] += Hs[tuple(p)]
        
    return output

def compute_best_partition_dtm(partitions_dict, max_size, min_size = 1):
    
    best_score = -np.inf
    best_partition = []
    
    for partition_str in partitions_dict.keys():
        
        partition = ast.literal_eval(partition_str)
        max_len = np.max([len(s) for s in partition])
        min_len = np.min([len(s) for s in partition])
        if max_len <= max_size and min_len >= min_size :
            
            if partitions_dict[partition_str] > best_score:
                best_score = partitions_dict[partition_str]
                best_partition = partition
                
    return best_partition


def model_selection_on_partitions_dtm(X, n_train, n_test, k, q=None, max_size=None):
    
    N, d = X.shape
    if max_size == None:
        max_size = d
    #Compute dictionary
    scores = compute_hat_Hs_dtm(X, n_train, n_test, max_size=max_size, k=k, q=q)
    #Compute partitions dictionary
    partitions_dict = compute_score_by_partition_dtm(X, max_size=max_size, Hs=scores)
    #best partitons
    best_partitions = {}

    for j in range(1, max_size+1):
        best = compute_best_partition_dtm(partitions_dict=partitions_dict, max_size=j, min_size=1)
        best_partitions[j] = {}
        best_partitions[j]['partition'] = best
        best_partitions[j]['score'] = partitions_dict[str(best)]
    
    return best_partitions, partitions_dict

def score_dtm_partition(x, y, partition, k=50,  q=None, correct_inf=False, normalize=True):
    
    n, d = y.shape
    density = np.ones(n)
    for p in partition:
        
        loc_density = dtm.DTMDensity(k=k, weights=None, q=q,
                                  n_jobs=-1, dim=None, normalize=normalize).fit(x[:, p]).transform(y[:, p])
        if correct_inf == True:
            loc_density[np.where(loc_density == np.inf)] = -np.inf
            loc_density[np.where(loc_density == -np.inf)] = np.max(loc_density)
        
        density *= loc_density
    
    return density

############################# Version with multiple k ############################

def compute_score_by_partition(X, max_size, Hs):
    
    N, d = X.shape
    lst = list(range(d))
    # Output : best partition for each max_size
    output = {}
    
    partitions = [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]
    keep = np.where(np.array([np.max([len(i) for i in p]) for p in partitions]) <=  max_size )[0]
    partitions_keep = [partitions[k] for k in keep]
    del partitions
    
    for partition in partitions_keep:
        output[str(partition)] = {}
        output[str(partition)]['score'] = 0
        ks = []
        for p in partition:
            output[str(partition)]['score'] += Hs[tuple(p)]['best_score']
            ks.append(Hs[tuple(p)]['best_k'])
        output[str(partition)]['ks'] = ks
        
    return output


def compute_hat_Hs(X, train_size, test_size, max_size, ks):
    
    N, d = X.shape
    Y = np.random.rand(10000, d)
    
    scores = {}
    X_train, X_test = train_test_split(X, train_size=train_size, test_size=test_size)
    
    sets = []
    for i in range(1, max_size+1):
        sets += list(itertools.combinations(range(d), i))
    
    for S in sets:
        scores[S] = {}
        loc_d = len(S)
        for k in ks:
            
            entropy = np.mean(np.log(score_dtm_partition(X_train, X_test, partition=[S], k=k,  q=None, correct_inf=False, normalize=True)))
            integral = 4 * score_dtm_partition(X_train, Y, partition=[S], k=k,  q=None, correct_inf=False, normalize=True)
            
            scores[S][k] = entropy - np.log(integral)
        ss = [scores[S][k] for k in ks]   
        best = np.where(ss==np.max(ss))[0][0]
    
        scores[S]["best_score"] = ss[best]
        scores[S]["best_k"] = ks[best]
            
    return scores

def model_selection_on_partitions(X, ks, n_train, n_test, max_size=None):
    
    N, d = X.shape
    
    if max_size == None:
        max_size = d
    #Compute dictionary
    scores = compute_hat_Hs(X, n_train, n_test, max_size=max_size, ks=ks)
    #Compute partitions dictionary
    partitions_dict = compute_score_by_partition(X, max_size=max_size, Hs=scores)
    #best partitons
    
    '''
    best_partitions = {}

    for j in range(1, max_size+1):
        best = compute_best_partition_dtm(partitions_dict=partitions_dict, max_size=j, min_size=1)
        best_partitions[j] = {}
        best_partitions[j]['partition'] = best
        best_partitions[j]['score'] = partitions_dict[str(best)]['score']
        best_partitions[j]['hs'] = partitions_dict[str(best)]['hs'] '''
    
    return partitions_dict, scores
