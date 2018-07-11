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

def autov(X):
	return LA.eig(np.cov(X,rowvar=False))

def mean(X):
	return np.mean(np.array(X), axis=0)

def variance(X):
	return np.var(np.array(X), axis=0)

def center(X, corr=False):
	if(corr):
		return (X - mean(X)) / variance(X)
	else:
		return (X - mean(X))

def decorr(X, PHI=None):
	if(PHI is None):
		_, PHI = autov(X)
	return np.dot(X, PHI)

def projo(X, vec):
	return np.dot(X, np.transpose(vec))


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

X = center(X, corr=False)
autovalores,autovetores = autov(X)

argmax = np.argpartition(-autovalores, 0)[:2]

X = np.dot(X, autovetores[:,[argmax[0],argmax[1]]])

# valor, vetor = LA.eig(X)
# print(valor)
# print(vetor)
