import numpy as np
import gudhi.point_cloud.knn as knn
import gudhi.point_cloud.dtm as dtm
import gudhi.clustering.tomato as tm
import itertools
import more_itertools as mit
import ast

from sklearn.model_selection import train_test_split

def compute_hat_Hs(X, train_size, test_size, max_size, min_k, max_k, n_Y):
    
    N, d = X.shape
    Y = np.random.rand(n_Y, d)
    
    ks = [i for i in range(min_k, max_k+1)]
    
    scores = {}
    scores['ks'] = ks
    entropies = {}
    entropies['ks'] = ks
    integrals = {}
    integrals['ks'] = ks
    X_train, X_test = train_test_split(X, train_size=train_size, test_size=test_size)
    
    sets = []
    for i in range(1, max_size+1):
        sets += list(itertools.combinations(range(d), i))
        
    
    for S in sets:
        
        scores[S] = {}
        entropies[S] = {}
        integrals[S] = {}
        loc_d = len(S)
        
        # Calculer les plus proches voisins pour k_max
        
        max_k = np.max(ks)

        nn_dists_test = knn.KNearestNeighbors(k=max_k, return_distance=True,
                                implementation='ckdtree', n_jobs=-1).fit(X_train[:, S]).transform(X_test[:, S])[1]
        nn_dists_Y = knn.KNearestNeighbors(k=max_k, return_distance=True,
                                implementation='ckdtree', n_jobs=-1).fit(X_train[:, S]).transform(Y[:, S])[1]

        
        test_distances_matrix = nn_dists_test[:, 0:max_k]**loc_d
        Y_distances_matrix = nn_dists_Y[:, 0:max_k]**loc_d

        density_test = 1 / np.cumsum(test_distances_matrix, axis=1)
        density_Y = 1 / np.cumsum(Y_distances_matrix, axis=1)

        entropies_vect = np.mean( np.log(density_test), axis=0)
        integrals_vect = np.mean( density_Y, axis=0 )
        
        for k in ks:
            scores[S][k] = entropies_vect[k-1] - np.log(integrals_vect[k-1])
            entropies[S][k] = entropies_vect[k-1]
            integrals[S][k] = integrals_vect[k-1]
        
        ss = [scores[S][k] for k in ks]
        best = np.where(ss==np.max(ss))[0][0]
    
        scores[S]["best_score"] = ss[best]
        scores[S]["best_k"] = ks[best]
            
    return scores, entropies, integrals


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

def compute_best_partition_dtm(partitions_dict, max_size, min_size = 1):
    
    best_score = -np.inf
    best_partition = []
    
    for partition_str in partitions_dict.keys():
        
        partition = ast.literal_eval(partition_str)
        max_len = np.max([len(s) for s in partition])
        min_len = np.min([len(s) for s in partition])
        if max_len <= max_size and min_len >= min_size :
            
            if partitions_dict[partition_str]['score'] > best_score:
                best_score = partitions_dict[partition_str]['score']
                best_partition = partition
                
    return best_partition

def model_selection_dtm(X, max_k, min_k, max_size, train_size, test_size, n_Y):
    
    scores_dict = compute_hat_Hs(X=X, train_size=train_size, test_size=test_size,
                                 max_size=max_size, min_k=min_k, max_k=max_k, n_Y=n_Y)[0]
    partitions_dict = compute_score_by_partition(X=X, max_size=d, Hs=scores_dict)
    best_partitions = {}

    for j in range(1, d+1):

        best = compute_best_partition_dtm(partitions_dict=partitions_dict, max_size=j, min_size = 1)
        best_partitions[j] = {}
        best_partitions[j]['partition'] = best
        best_partitions[j]['score'] = partitions_dict[str(best)]['score']
        best_partitions[j]['ks'] = partitions_dict[str(best)]['ks']
        print(best_partitions[j])
    
    return scores_dict, partitions_dict, best_partitions
