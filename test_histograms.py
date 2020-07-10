import numpy as np
from sklearn.preprocessing import MinMaxScaler

from includes.structured_histograms import *

alpha = 0.8
beta = 0.7

cov = np.array( [[1, alpha, 0, 0],
                 [alpha, 1, 0, 0],
                 [0, 0, 1, beta],
                 [0, 0, beta, 1]] )

# alpha = 0.95
# beta = 0.95
# gamma = 0.95

# cov = np.array( [[1, 0, 0, 0],
#                  [0, 1, alpha, gamma],
#                  [0, alpha, 1, beta],
#                  [0, gamma, beta, 1]] )

# mean = np.zeros(4)


N = 100000
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


m = 5

for p in partitions:

    a = Regular_Histogram(p, m)
    a.fit(X)
    print(p)
    print(a.test_log_likelihood(X_test))
