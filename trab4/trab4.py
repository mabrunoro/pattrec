#!/usr/bin/env python3

import sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
#from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

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

FEATNAME = XMV + XMEAS

AFUN = 'relu'
#AFUN = 'sigmoid'
NUMHIDDEN = 3
ACTHIDDEN ='sigmoid'
ACTOUT = 'softmax'

OPTIMIZER = SGD()
LOSS='categorical_crossentropy'


KSIZE = 3    # size of the convolution filter (image would be e.g. tupel (3,3) )
FILTERS = 3 # number of convolution FILTERS

FOLDS = 10
EPOCHS = 30

#FOLDS = 2
#EPOCHS = 2

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


# class browser: pydoc -g keras
def _buildsimplemodel(d, chan, numclasses, KSIZE, FILTERS ):

    print('buildsimplemodel>\n\tnum_classes=', numclasses, 'dim=', d, 'channels=', chan)

    model = Sequential()

    # create 1-D convolution layer
    # https://keras.io/layers/convolutional/
    #
    # input: dx1-dimensional signals with chan channels -> (d, chan) tensors.
    input_shape = (d, chan)

    # Output Shape: (None, d-KSIZE+1, FILTERS) = (d1,d2,d3)
    # Param # = (chan x KSIZE + 1) x FILTERS  ; (+0, if no bias)
    convlay = Conv1D(name='convlay', input_shape=input_shape, kernel_size=KSIZE, filters=FILTERS, use_bias=True)
    model.add(convlay)
    # Output Shape: (None, d-KSIZE+1, FILTERS) = (d1,d2,d3)
    actlay = Activation(AFUN)
    model.add(actlay)   # Function of ReLU activation is detector, [1], p. 71, fig. 5.9

    # create 1-D pooling layer
    # https://keras.io/layers/pooling/
    # Pooling operation: [1] p. 68
    # Output Shape: (None, trunc(d2/pool_size), d3) = (d1,d4,d3)
    #poollay = MaxPooling1D(pool_size=4, strides=None, padding='valid')
    #model.add(poollay)

    # now arrived at output of Convolution-Detector-Pooling Building Block, [1], p.70

    # Output Shape: (None, d4 x d3) = (d1,d5)
    model.add(Flatten())

    # Output Shape: (None, NUMHIDDEN1) = (d1,d6)
    # Param # = (d5 + 1) x d6  ; (+0, if no bias)
    model.add(Dense(NUMHIDDEN, activation=ACTHIDDEN, name='hidlay'))
    #model.add(Dropout(0.5))

    # Output Shape: (None, numclasses) = (d1,d7)
    # Param # = (d6 + 1) x d7  ; (+0, if no bias)
    model.add(Dense(numclasses, activation=ACTOUT))

    # https://keras.io/OPTIMIZERs/#sgd
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
    model.compile(loss=LOSS, optimizer=OPTIMIZER)

    print(model.summary())
    #config = model.get_config()
    #print config

    return model




def CNN(X, y, datadir, classname):
	#print 'CNN>\n'

	print('Classifier: Convolutional Network\n')
	# print('class browser: pydoc -g keras\n')
	n_samples, d = X.shape[:2]

	num_classes = len(classname)
	Y = to_categorical(y, num_classes=num_classes)

	skf = StratifiedKFold(n_splits=FOLDS, shuffle=True)

	y_pred_overall = []
	y_test_overall = []

	y = np.resize(y,(n_samples,))

	# The model CANNOT be build here, since for each fold, the weights must be reset
	#model = _buildsimplemodel(d, chan, num_classes, KSIZE, FILTERS )
	#model = _buildmodel(d, chan, num_classes, KSIZE, FILTERS )

	# split does not work for 3-D array, only need y for split
	dummyX = np.zeros(n_samples)

	# seed = 42
	# test_size = 0.3
	# train_index, test_index = train_test_split(y, shuffle=True, test_size=test_size, random_state=seed)
	#print(train_index, test_index)

	# print(X.shape, y.shape, dummyX.shape)
	# exit(0)

	k = 1
	for train_index, test_index in skf.split(dummyX, y): # if K-fold
	    #print "TRAIN:", train_index, "TEST:", test_index

	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]
	    Y_train, Y_test = Y[train_index], Y[test_index]
	    print('X_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)

	    model = _buildsimplemodel(d, 1, num_classes, KSIZE, FILTERS )
	    #model.compile(loss=LOSS,OPTIMIZER=OPTIMIZER) #Must reset weights

	    model.fit(X_train, Y_train, batch_size=1, epochs=EPOCHS, verbose=1)

	    w = model.get_layer(name='convlay').get_weights()

	    loss_score = model.evaluate(X_test, Y_test, batch_size=1)
	    print('\t$$$ Fold=', k, 'of ', FOLDS, ' loss_score=', loss_score)
	    y_pred_class = model.predict_classes(X_test, batch_size=1, verbose=2)


	    y_pred_overall = np.concatenate([y_pred_overall, y_pred_class])
	    y_test_overall = np.concatenate([y_test_overall, y_test])
	    k += 1

	print('Hidden layer:')
	print(model.get_layer(name='hidlay').get_weights())
	print('CNN Classification Report:')
	print(classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
	print('CNN Confusion Matrix:')
	print(confusion_matrix(y_test_overall, y_pred_overall))
	print('CNN Classification Report:')
	print(classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
	print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
	print('Macro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='macro')))
	print('Micro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='micro')))
	print('CNN Confusion Matrix:')
	print(confusion_matrix(y_test_overall, y_pred_overall))


def main(filename='all.csv'):
	X, Y, y, labels, classes = csvread(filename=filename)

	classlabels = np.unique(classes)

	# print(X.shape, Y.shape, len(y), classlabels.shape)
	# nattr = X.shape[1]
	# nsamp = X.shape[0]

	# samples = random.sample(range(nsamp), nsamp)	# obtém as amostras de forma aleatória

	# ntreino = math.floor(0.75 * nsamp)
	# nteste = nsamp - ntreino

	X = np.reshape(X, X.shape + (1,))

	CNN(X, Y, './', classlabels)


if __name__ == '__main__':
	main()

	# classname = ['class1', 'class2', 'class3']
	# c = len(classname)   # classes
	#
	# # Generate dummy data
	# n = 50 # samples
	# chan = 2    # channels e.g. RGB in image
	# d = 64 # signal features
	#
	# x_train = np.random.random((n, d, chan))
	# y_train = np.random.randint(c, size=(n, 1)) # 1-D array
	# #y_train = keras.utils.to_categorical(y_train, num_classes=c) # 1-out-of-c
	# x_test = np.random.random((n, d, chan))
	# y_test = np.random.randint(c, size=(n, 1))
	# #y_test = keras.utils.to_categorical(y_test, num_classes=c)
	#
	# #print 'shape training patterns=', x_train.shape, '\nshape training labels=', y_train.shape
	# #print x_train[0,0:5,0]  # first pattern, first five features
	# #print y_train[0]        # first pattern, one-out-of-c label
	#
	# X = np.concatenate((x_train,x_test),axis=0)
	# y = np.concatenate((y_train,y_test),axis=0)
	# '''
	# print 'shape X=', X.shape, '\nshape y=', y.shape
	# print '\nx_train[0]=\n', x_train[0], '\ny_train[0]=\n', y_train[0]
	# print '\nX[0]=\n', X[0], '\ny[0]=\n', y[0]
	# '''
	# KSIZE = 32    # size of the convolution filter (image would be e.g. tupel (3,3) )
	# FILTERS = 2 # number of convolution FILTERS
	#
	# CNN( X, y, './', classname )
