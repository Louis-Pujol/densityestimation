import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from kde import *
import itertools
import more_itertools as mit
import fcmdata_helpers.metrics.clustering as metrics
import gudhi.clustering.tomato as tm
import ast
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def compute_hat_Hs(X, train_size, test_size, max_size, hs):
    
    N, d = X.shape
    
    scores = {}
    X_train, X_test = train_test_split(X, train_size=train_size, test_size=test_size)
    
    sets = []
    for i in range(1, max_size+1):
        sets += list(itertools.combinations(range(d), i))
    
    for S in sets:
        scores[S] = {}
        loc_d = len(S)
        for h in hs:
            scores[S][h] = np.mean(KernelDensity(X=X_train[:, S],
                                    bw=h*np.ones(loc_d)).score(x=X_test[:, S], method='gpu',
                                                                n_jobs=-1, log=True))
        ss = [scores[S][h] for h in hs]   
        best = np.where(ss==np.max(ss))[0][0]
    
        scores[S]["best_score"] = ss[best]
        scores[S]["best_h"] = hs[best]
            
    return scores


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
        hs = []
        for p in partition:
            output[str(partition)]['score'] += Hs[tuple(p)]['best_score']
            hs.append(Hs[tuple(p)]['best_h'])
        output[str(partition)]['hs'] = hs
        
    return output

def compute_best_partition(partitions_dict, max_size, min_size = 1):
    
    best_score = -np.inf
    best_partition = []
    
    for partition_str in partitions_dict.keys():
        
        partition = ast.literal_eval(partition_str)
        max_len = np.max([len(s) for s in partition])
        min_len = np.min([len(s) for s in partition])
        if max_len <= max_size and min_len >= min_size :
            
            if partitions_dict[partition_str]["score"] > best_score:
                best_score = partitions_dict[partition_str]["score"]
                best_partition = partition
                
    return best_partition


def model_selection_on_partitions(X, hs, n_train, n_test, max_size=None):
    
    N, d = X.shape
    if max_size == None:
        max_size = d
    #Compute dictionary
    scores = compute_hat_Hs(X, n_train, n_test, max_size=max_size, hs=hs)
    #Compute partitions dictionary
    partitions_dict = compute_score_by_partition(X, max_size=max_size, Hs=scores)
    #best partitons
    best_partitions = {}

    for j in range(1, max_size+1):
        best = compute_best_partition(partitions_dict=partitions_dict, max_size=j, min_size=1)
        best_partitions[j] = {}
        best_partitions[j]['partition'] = best
        best_partitions[j]['score'] = partitions_dict[str(best)]['score']
        best_partitions[j]['hs'] = partitions_dict[str(best)]['hs']
    
    return best_partitions, partitions_dict

def correction_bandwidth(X, partition, Ns, n_test, hs, M=5, savefigure=False, path = ""):
    
    correction = {}
    
    for S in partition:
        correction[str(S)] = {}
        best_h = []
        loc_d = len(S)
        for N1 in Ns:
            
            loc_best_h = 0
            for j in range(M):
                X_train, X_test = train_test_split(X, train_size=N1, test_size=n_test)
                a = [np.mean(KernelDensity(X=X_train[:, S], bw= h*np.ones(loc_d)).score(x=X_test[:, S], method='gpu', log=True))
                for h in hs]
                loc_best_h += (1 / M) * hs[np.where(a==np.max(a))[0][0]]
            best_h.append(loc_best_h)
            
        plt.figure(figsize=(10,5))
        
        plt.subplot(121)
        plt.plot(Ns, best_h)
        plt.title(S)
        plt.xlabel("n_train")
        plt.ylabel("h_opt")

        reg = LinearRegression().fit(np.log(Ns).reshape(-1, 1), np.log(best_h))
        a = reg.coef_[0]
        b = reg.intercept_
        plt.subplot(122)
        plt.plot(np.log(Ns), np.log(best_h))
        plt.plot([np.log(Ns[0]), np.log(Ns[-1])], [a*np.log(Ns[0]) + b, a*np.log(Ns[-1]) + b])
        plt.xlabel("log(n_train)")
        plt.ylabel("log(h_opt)")
        plt.title("log(h_opt) = " + str(round(a, 2)) + "log(n_train) + " + str(round(b, 2)))
        plt.tight_layout()
        
        if savefigure == False:
            plt.show()
        else:
            plt.savefig(path+"correction"+str(S)+".png")
            plt.clf()
        correction[str(S)]["C"] = np.exp(b)
        correction[str(S)]["alpha"] = -a
    
    return correction

def compute_density_with_partition(X, partition, correction=None, hs=None, log=True):
    #Soit on fournit hs, soit la fonction correction
    
    N, d = X.shape
    if correction != None:
        
        density = np.zeros(N)
        for S in partition:
            loc_d = len(S)
            h = correction[str(S)]['C'] * ( N ** (-correction[str(S)]['alpha']) )
            density += KernelDensity(X=X[:, S], bw=h*np.ones(loc_d)).score(x=X[:, S],
                                     method='gpu', log=log)
        
        return density
    
    elif hs != None:
        
        density = np.zeros(N)
        for i in range(len(partition)):
            S = partition[i]
            loc_d = len(S)
            h = hs[i]
            density += KernelDensity(X=X[:, S], bw=h*np.ones(loc_d)).score(x=X[:, S],
                                     method='gpu', log=log)
        return density

    else:
        print("you must specify correction or hs")
        
def show_correction_improvement(X, partition, Ns, n_test, correction, hs, M=5):
    
    scores_corrected = []
    scores_notcorrected = []

    for n_train in Ns:
        
        loc_score_corrected = 0
        loc_score_notcorrected = 0
        for k in range(M):
            
            X_train, X_test = train_test_split(X, train_size=n_train, test_size=n_test)
            
            for i in range(len(partition)):
                S = partition[i]
                loc_d = len(S)
                h_notcorrected = hs[i]
                h_corrected = correction[str(S)]['C'] * ( n_train ** (-correction[str(S)]['alpha']) )
                
                loc_score_corrected += np.mean(KernelDensity(X=X_train[:, S],
                                                             bw=h_corrected*np.ones(loc_d)).score(x=X_test[:, S],
                                                                                                    method='gpu',
                                                                                                    log=True)) / M
                loc_score_notcorrected += np.mean(KernelDensity(X=X_train[:, S],
                                                             bw=h_notcorrected*np.ones(loc_d)).score(x=X_test[:, S],
                                                                                                    method='gpu',
                                                                                                    log=True)) / M
        scores_corrected.append(loc_score_corrected)
        scores_notcorrected.append(loc_score_notcorrected)
            
    plt.plot(Ns, scores_notcorrected, label="without correction")
    plt.plot(Ns, scores_corrected, label="with correction")
    plt.legend()
    plt.xlabel("n_train")
    plt.ylabel("score")
    plt.show()
