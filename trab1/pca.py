from __future__ import print_function
"""
========================================
Principal Component Analysis
========================================

"""
print(__doc__)

# -*- coding: utf-8 -*-


import sklearn.datasets as datasets
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


# Get iris data
iris = datasets.load_iris()
X = iris.data
labels = iris.target
classlabels = np.unique(iris.target)
classes = iris.target_names
featname = iris.feature_names


feat1 = 2   # feature #3
feat2 = 3   # feature #4
X = X[:, [feat1, feat2]]   # Take only features 3 and 4
X = X[[0, 9, 52, 56, 104, 148], :] # Take six random samples

print('Dataset X=\n', X)

# Plot the data points
# plt.scatter(X[:, 0], X[:, 1], c='black', cmap=plt.cm.Set1, edgecolor='k', label='data')
# plt.xlabel(featname[feat1])
# plt.ylabel(featname[feat2])
# plt.legend()
# plt.show()

np.cov(X, rowvar=False)

# valor, vetor = LA.eig(X)
# print(valor)
# print(vetor)
