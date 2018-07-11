#!/usr/bin/env python3

# Author: Marco Thome

import sys
import sklearn.datasets as datasets
import matplotlib.pyplot as pyplot
import numpy as np
import math

def show(X):
	for i in X:
		print(i)

def plot(X,title=''):
	show(X)
	if((len(X.shape) > 1) and (X.shape[1] > 2)):
		pyplot.scatter(X, [list(range(len(X))) for i in range(len(X[0]))])
	elif((len(X.shape) > 1) and (X.shape[1] == 2)):
		pyplot.scatter(X[:,0], X[:,1])
		pyplot.ylim(math.floor(min(X[:,1])),math.ceil(max(X[:,1])))
	else:
		pyplot.scatter(X,np.zeros(len(X)))
	pyplot.title(title)
	pyplot.show()

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

def main(infile=None):
	if(infile is not None):
		with open(infile) as f:
			data = f.read().split()
		x = []
		for i in data:
			x.append(list(map(float,i.split(',')[:4])))
	else:
		x = np.array(datasets.load_iris().data)
	X = x[[0,9,52,56,104,148],2:4]
	# X = []
	# for i in [0,9,52,56,104,148]:
	# 	X.append(x[i][2:4])
	# X = np.array(X)

	print('Matriz')
	plot(X,'Matriz de Dados')


	print('\nCovariância')
	print(np.cov(X, rowvar=False))


	autovalores, autovetores = autov(X)
	print('\nAutovalores')
	print(autovalores)
	print('\nAutovetores')
	print(autovetores)


	print('\nCentralizando')
	Y = center(X)
	plot(Y,'Dados Centralizados')


	print('\nDecorrelacionando')
	Y = decorr(Y, PHI=autovetores)
	plot(Y, 'Dados Decorrelacionados')


	print('\nPrimeira Componente Principal')
	argmax = np.argmax(autovalores)
	prin = autovetores[:,argmax]
	print(prin)


	print('\nProjeção')
	fc = projo(X, prin)
	plot(fc,'Projeção')
	# print(fc)


	print('\nVariância e Maior Autovalor de X')
	print(np.cov(fc))
	print(autovalores[argmax])

if(__name__ == '__main__'):
	if(len(sys.argv) > 1):
		main(sys.argv[1])
	else:
		main()
