import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from includes.structured_histograms import *

alpha = 0.7
beta = 0.8

cov = np.array( [[1, alpha, 0, 0],
                 [alpha, 1, 0, 0],
                 [0, 0, 1, beta],
                 [0, 0, beta, 1]] )

#alpha = 0.95
#beta = 0.95
#gamma = 0.95

#cov = np.array( [[1, 0, 0, 0],
                 #[0, 1, alpha, gamma],
                 #[0, alpha, 1, beta],
                 #[0, gamma, beta, 1]] )

mean = np.zeros(4)


N = 1000
mean = np.zeros(4)

X = np.random.multivariate_normal(mean, cov, N)
a = Regular_Histogram([[0, 1], [2] ,[3]], 20)
a.fit(X)
a.complete_histogram()


N = 1000
mean = np.zeros(4)


X = np.random.multivariate_normal(mean, cov, N)
X = MinMaxScaler().fit_transform(X)

X_test = np.random.multivariate_normal(mean, cov, N)
X_test = MinMaxScaler().fit_transform(X_test)


a = Regular_Histogram([[0], [1], [2] ,[3]], 20)
a.fit(X)
a.test_log_likelihood(X_test)

partitions = [[0,1,2,3],
              [[0],[1,2,3]],
              [[1],[0,2,3]],
              [[2],[0,1,3]],
              [[3],[0,1,2]],
              [[0,1],[2,3]],
              [[0,2],[1,3]],
              [[0,3],[1,2]]]


m = 20

for p in partitions:

    a = Regular_Histogram(p, m)
    a.fit(X)
    print(p)
    print(a.test_log_likelihood(X_test, opt=2, lambd=1))
    
    

#def add_one_to_str(a):
    
    #t = [str(i) for i in range(10)]
    #b = ""
    #for i in range(len(a)):
        #if a[i] in t:
            #b += str( int(a[i]) + 1 )
        #else:
            #b += a[i]
    #return b

#alpha = 0.8
#beta = 0.7

#cov = np.array( [[1, alpha, 0, 0],
                 #[alpha, 1, 0, 0],
                 #[0, 0, 1, beta],
                 #[0, 0, beta, 1]] )

#mean = np.zeros(4)

#X = np.random.multivariate_normal(mean, cov, 1000)
#X = MinMaxScaler().fit_transform(X)
#print(X.shape)

#plt.scatter(X[:, 0], X[:, 1], s=0.5)
#plt.show()
#plt.scatter(X[:, 0], X[:, 2], s=0.5)
#plt.show()





#partitions = [[0,1,2,3],
              #[[0],[1,2,3]],
              #[[1],[0,2,3]],
              #[[2],[0,1,3]],
              #[[3],[0,1,2]],
              #[[0,1],[2,3]],
              #[[0,2],[1,3]],
              #[[0,3],[1,2]]]


#m = 20
#K = 100
#n = 1000


#2x2
#results = {}
#for p in partitions:
    #if len(p)==2:
        #if np.max([len(p[0]), len(p[1])]) == 2:
            #results[str(p)] = []

#for i in range(K):

    #X = np.random.multivariate_normal(mean, cov, n)
    #X = MinMaxScaler().fit_transform(X)

    #for p in partitions:
        #if len(p)==2:
            #if np.max([len(p[0]), len(p[1])]) == 2:
                
                #a = Regular_Histogram(p, m)
                #a.fit(X)
                
                #results[str(p)].append(a.log_likelihood)
    
#labels, data = [*zip(*results.items())]  # 'transpose' items to parallel key, value lists
#labels2 = ()
#for i in range(len(labels)):
    #labels2 += (add_one_to_str(labels[i]),)
#plt.boxplot(data)
#plt.xticks(range(1, len(labels2) + 1), labels2)
#plt.title("alpha = " + str(alpha) + ", beta = " +str(beta))
#plt.ylabel("log-likelihood (train)")
#plt.show()
#plt.clf()


#m = 10
#K = 10
#n = 1000


#2x2
#results = {}
#for p in partitions:
    #results[str(p)] = []
            

#for i in range(K):
    
    #print(i)

    #X = np.random.multivariate_normal(mean, cov, n)
    #X = MinMaxScaler().fit_transform(X)
    
    #X_test = np.random.multivariate_normal(mean, cov, n)
    #X_test = MinMaxScaler().fit_transform(X_test)

    #for p in partitions:
                
        #a = Regular_Histogram(p, m)
        #a.fit(X)
                
        #results[str(p)].append(a.test_log_likelihood(X_test))
    
#labels, data = [*zip(*results.items())]  # 'transpose' items to parallel key, value lists
#labels2 = ()
#for i in range(len(labels)):
    #labels2 += (add_one_to_str(labels[i]),)
#plt.boxplot(data)
#plt.xticks(range(1, len(labels2) + 1), labels2)
#plt.title("alpha = " + str(alpha) + ", beta = " +str(beta))
#plt.ylabel("log-likelihood (test)")
#plt.show()
#plt.clf()
