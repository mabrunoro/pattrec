from __future__ import print_function
"""
========================================
Linear Machine on the iris dataset
========================================

Plot decision surface of multi-class SGD on iris dataset.
Stochastic Gradient Descent
http://scikit-learn.org/stable/modules/sgd.html
The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
are represented by the dashed lines.

"""
# print(__doc__)

# -*- coding: utf-8 -*-



import numpy as np
import sys


import csv

filename = '../out/all.csv'
delimiter = '\t'

def csvread(filename, delimiter = '\t'):
	if(sys.version_info[0] == 2):	# Python 2
	    f = open(filename, 'rb')
	else:							# Python 3
		f = open(filename, 'rt', encoding='utf-8')
	reader = csv.reader(f, delimiter=delimiter)
	ncol = len(next(reader)) # Read first line and count columns
	nfeat = ncol-1
	f.seek(0)              # go back to beginning of file
	#print('ncol=', ncol)

	x = np.zeros(nfeat)
	X = np.empty((0, nfeat))
	y = []
	for row in reader:
	    #print(row)
	    for j in range(nfeat):
	        x[j] = float(row[j])
	        #print('j=', j, ':', x[j])
	    X = np.append(X, [x], axis=0)
	    label = row[nfeat]
	    y.append(label)
	    #print('label=', label)
	# print('X.shape=\n', X.shape, '\nX=\n', X,'\n')
	# print('y=\n', y)


	# Resubsitution for all methods
	from sklearn.preprocessing import LabelBinarizer, LabelEncoder
	lb = LabelBinarizer()
	Y = lb.fit_transform(y)
	classname = lb.classes_
	# print('lb.classes_=', lb.classes_, '\nY=\n',Y)

	le = LabelEncoder()
	ynum = le.fit_transform(y)
	# print(ynum)
	f.close()
	# sys.exit(0)
	return X, Y, y, ynum, classname


X, Y, y, ynum, classname = csvread(filename=filename, delimiter=delimiter)

XMV = ['D feed flow (stream 2)',
    'E feed flow (stream 3)',
    'A feed flow (stream 1)',
    'A and C feed flow (stream 4)',
    'Compressor recycle valve',
    'Purge valve (stream 9)',
    'Separator pot liquid flow (stream 10)',
    'Stripper liquid product flow (stream 11)',
    'Stripper steam valve',
    'Reactor cooling water flow',
    'Condenser cooling water flow']
XMEAS = ['Input Feed - A feed (stream 1)',
    'Input Feed - D feed (stream 2)',
    'Input Feed - E feed (stream 3)',
    'Input Feed - A and C feed (stream 4)',
    'Reactor feed rate (stream 6)',
    'Reactor pressure',
    'Reactor level',
    'Reactor temperature',
    'Separator - Product separator temperature',
    'Separator - Product separator level',
    'Separator - Product separator pressure',
    'Separator - Product separator underflow (stream 10)',
    'Stripper level',
    'Stripper pressure',
    'Stripper underflow (stream 11)',
    'Stripper temperature',
    'Stripper steam flow',
    'Miscellaneous - Recycle flow (stream 8)',
    'Miscellaneous - Purge rate (stream 9)',
    'Miscellaneous - Compressor work',
    'Miscellaneous - Reactor cooling water outlet temperature',
    'Miscellaneous - Separator cooling water outlet temperature',
    'Reactor Feed Analysis - Component A',
    'Reactor Feed Analysis - Component B',
    'Reactor Feed Analysis - Component C',
    'Reactor Feed Analysis - Component D',
    'Reactor Feed Analysis - Component E',
    'Reactor Feed Analysis - Component F',
    'Purge gas analysis - Component A',
    'Purge gas analysis - Component B',
    'Purge gas analysis - Component C',
    'Purge gas analysis - Component D',
    'Purge gas analysis - Component E',
    'Purge gas analysis - Component F',
    'Purge gas analysis - Component G',
    'Purge gas analysis - Component H',
    'Product analysis -  Component D',
    'Product analysis - Component E',
    'Product analysis - Component F',
    'Product analysis - Component G',
    'Product analysis - Component H']

featname = XMV + XMEAS


labels = ynum
classes = classname
classlabels = np.unique(ynum)


feat1 = 49 # First feature
feat2 = 12 # Second feature
X2feat = X[:, [feat1,feat2]] # only the first two features


'''
import sklearn.datasets as datasets
# Get iris data
iris = datasets.load_iris()
X = iris.data
labels = iris.target
classlabels = np.unique(iris.target)
classes = iris.target_names
featname = iris.feature_names


feat1 = 2 # First feature
feat2 = 3 # Second feature
X2feat = iris.data[:, [feat1,feat2]] # only the first two features
y = iris.target
ynum = y
'''




import matplotlib.pyplot as plt

# X = X2feat
# y = ynum
colors = "bry"
#
#
# # standardize
# mean = X.mean(axis=0)
# std = X.std(axis=0)
# print('mean=', mean, 'std=', std)
# X = (X - mean) / std
#
#
# # Plot also the training points
# for i, color in zip(classlabels, colors):
# #for i, color in zip(clf.classes_, colors):
#     idx = np.where(y == i)
#     plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classes[i],
#                 cmap=plt.cm.Paired, edgecolor='black', s=20)
# plt.title('Tennessee Eastman: Classes in Feature Space')
# plt.axis('tight')
#
# # Plot the three one-against-all classifiers
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
#
# plt.xlabel(featname[feat1])
# plt.ylabel(featname[feat2])
#
# plt.legend()
# plt.show()
#
# sys.exit(0)

def autov(X):
	return np.linalg.eig(np.cov(X,rowvar=False))

def mean(X):
	return np.mean(np.array(X), axis=0)

def center(X):
	return (X - mean(X))

def decorr(X, PHI=None):
	if(PHI is None):
		_, PHI = autov(X)
	return np.dot(X, PHI)

def projo(X, vec):
	return np.dot(X, np.transpose(vec))

autovalores, autovetores = autov(X)

Y = center(X)
Y = decorr(Y, PHI=autovetores)

argmax = np.argpartition(-autovalores, 3)[:3]
# print('Autovalores')
# print(autovalores[argmax])
# print('Autovetores')
# print(autovetores[argmax])

proj = projo(X, autovetores[argmax[:3]])
featnames = np.array(list(featname[i] for i in argmax[:3]))

import math
from mpl_toolkits.mplot3d import Axes3D

def show(X):
	for i in X:
		print(i)

def plot(X,title='', axlab=['','','']):
	# show(X)
	# 2D plot
	if((len(X.shape) > 1) and (X.shape[1] == 2)):
		for (i, color) in zip(classlabels, colors):
			idx = np.where(ynum == i)
			plt.scatter(X[idx,0], X[idx,1], c=color, label=classes[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
		plt.ylim(math.floor(min(X[:,1])),math.ceil(max(X[:,1])))
		plt.xlabel(axlab[0])
		plt.ylabel(axlab[1])
	# 3D plot
	elif((len(X.shape) > 1) and (X.shape[1] == 3)):
		fig = plt.figure()
		ax = Axes3D(fig)
		for (i, color) in zip(classlabels, colors):
			idx = np.where(ynum == i)
			ax.scatter(X[idx,0], X[idx,1], X[idx,2], c=color, label=classes[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
		ax.set_xlabel(axlab[0])
		ax.set_ylabel(axlab[1])
		ax.set_zlabel(axlab[2])
	# +4D plot
	elif((len(X.shape) > 1) and (X.shape[1] > 2)):
		plt.scatter(X, [list(range(len(X))) for i in range(len(X[0]))])
	# 1D plot
	else:
		print('Printing vector')
		plt.scatter(X,np.zeros(len(X)))
	plt.title(title)
	plt.legend()
	plt.show()

# print('Featnames:',featnames)
print('Variances (3 largest):\n', autovalores[:3],'\n')
print('Total variance 2D:\n', np.sum(autovalores[:2]))
plot(proj[:,:2], title='PCA 2D', axlab=featnames)
print('Total variance 3D:\n', np.sum(autovalores[:3]))
plot(proj, title='PCA 3D', axlab=featnames)
