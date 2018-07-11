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


afun = 'relu'
#afun = 'sigmoid'
numhidden = 50
activationhidden ='sigmoid'
activationout = 'softmax'

optimizer = SGD()
loss='categorical_crossentropy'


kernel_size = 3    # size of the convolution filter (image would be e.g. tupel (3,3) )
filters = 3 # number of convolution filters

folds = 10
epochs = 30

#folds = 2
#epochs = 2


# class browser: pydoc -g keras
def _buildsimplemodel(d, chan, numclasses, kernel_size, filters ):

    # print 'buildsimplemodel>\n\tnum_classes=', numclasses, 'dim=', d, 'channels=', chan

    model = Sequential()

    # create 1-D convolution layer
    # https://keras.io/layers/convolutional/
    #
    # input: dx1-dimensional signals with chan channels -> (d, chan) tensors.
    input_shape = (d, chan)

    # Output Shape: (None, d-kernel_size+1, filters) = (d1,d2,d3)
    # Param # = (chan x kernel_size + 1) x filters  ; (+0, if no bias)
    convlay = Conv1D(name='convlay', input_shape=input_shape,\
                     kernel_size=kernel_size, filters=filters, use_bias=True)
    model.add(convlay)
    # Output Shape: (None, d-kernel_size+1, filters) = (d1,d2,d3)
    actlay = Activation(afun)
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

    # Output Shape: (None, numhidden1) = (d1,d6)
    # Param # = (d5 + 1) x d6  ; (+0, if no bias)
    model.add(Dense(numhidden, activation=activationhidden))
    #model.add(Dropout(0.5))

    # Output Shape: (None, numclasses) = (d1,d7)
    # Param # = (d6 + 1) x d7  ; (+0, if no bias)
    model.add(Dense(numclasses, activation=activationout))

    # https://keras.io/optimizers/#sgd
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
    model.compile(loss=loss,optimizer=optimizer)

    print(model.summary())
    #config = model.get_config()
    #print config

    return model


# class browser: pydoc -g keras
def _buildmodel(d, chan, numclasses, kernel_size, filters ):

    # print 'buildmodel>\n\tnum_classes=', numclasses, 'dim=', d, 'channels=', chan

    model = Sequential()

    # create 1-D convolution layer
    # https://keras.io/layers/convolutional/
    #
    # input: dx1-dimensional signals with chan channels -> (d, chan) tensors.
    input_shape = (d, chan)

    # Output Shape: (None, d-kernel_size+1, filters) = (d1,d2,d3)
    # Param # = (chan x kernel_size + 1) x filters  ; (+0, if no bias)
    convlay = Conv1D(name='convlay', input_shape=input_shape,\
                     kernel_size=kernel_size, filters=filters, use_bias=True)
    model.add(convlay)
    # Output Shape: (None, d-kernel_size+1, filters) = (d1,d2,d3)
    afun = 'relu'
    #afun = 'sigmoid'
    actlay = Activation(afun)
    model.add(actlay)   # Function of ReLU activation is detector, [1], p. 71, fig. 5.9

    # create 1-D pooling layer
    # https://keras.io/layers/pooling/
    # Pooling operation: [1] p. 68
    # Output Shape: (None, trunc(d2/pool_size), d3) = (d1,d4,d3)
    poollay = MaxPooling1D(pool_size=4, strides=None, padding='valid')
    model.add(poollay)

    # now arrived at output of Convolution-Detector-Pooling Building Block, [1], p.70

    # Output Shape: (None, d4 x d3) = (d1,d5)
    model.add(Flatten())

    numhidden1 = 50
    # Output Shape: (None, numhidden1) = (d1,d6)
    # Param # = (d5 + 1) x d6  ; (+0, if no bias)
    model.add(Dense(numhidden1, activation='relu'))
    model.add(Dropout(0.5))

    # Output Shape: (None, numclasses) = (d1,d7)
    # Param # = (d6 + 1) x d7  ; (+0, if no bias)
    model.add(Dense(numclasses, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print(model.summary())
    #config = model.get_config()
    #print config
    from keras.utils.vis_utils import plot_model
    modelfilename = 'CNN.png'
    # print 'Dumping model to ', modelfilename
    plot_model(model, to_file=modelfilename, show_shapes=True)

    return model


def CNN( X, y, datadir, classname ):
    #print 'CNN>\n'

    # print 'Classifier: Convolutional Net\n'
    # print 'class browser: pydoc -g keras\n'
    n_samples, d, chan = X.shape

    num_classes = len(classname)
    Y = to_categorical(y, num_classes=num_classes)

    skf = StratifiedKFold(n_splits=folds, shuffle=True)

    y_pred_overall = []
    y_test_overall = []

    y = np.resize(y,(n_samples,))

    # The model CANNOT be build here, since for each fold, the weights must be reset
    #model = _buildsimplemodel(d, chan, num_classes, kernel_size, filters )
    #model = _buildmodel(d, chan, num_classes, kernel_size, filters )

    # split does not work for 3-D array, only need y for split
    dummyX = np.zeros(n_samples)


    seed = 42
    test_size = 0.3
    train_index, test_index = train_test_split(y, shuffle=True,
                                    test_size=test_size, random_state=seed)
    #print(train_index, test_index)

    k = 1
    for train_index, test_index in skf.split(dummyX, y): # if K-fold
    #for i in range(1):    # if simple hold-out split
        #print "TRAIN:", train_index, "TEST:", test_index

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        print('X_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)

        model = _buildsimplemodel(d, chan, num_classes, kernel_size, filters )
        #model.compile(loss=loss,optimizer=optimizer) #Must reset weights
        #w = model.get_layer(name='convlay').get_weights() #DEBUG
        #print '\n\nBefore fit', (w[0])[0,0,0] #DEBUG
        model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=1)

        w = model.get_layer(name='convlay').get_weights()
        #print 'd=',d,'chan=',chan,'num_classes=', num_classes, 'kernel_size=',kernel_size, 'filters=',filters
        #print 'After fit ', (w[0])[0,0,0] #DEBUG

        loss_score = model.evaluate(X_test, Y_test, batch_size=1)
        # print '\n$$$ Fold=', k, 'of ', folds, ' loss_score=', loss_score
        #Y_pred = model.predict(X_test,batch_size=1, verbose=2)
        y_pred_class = model.predict_classes(X_test, batch_size=1, verbose=2)
        #print '\ny_train[0]', y_train[0], '\nY_train[0]', Y_train[0]
        #print 'y_pred_class[0]', y_pred_class[0]

        y_pred_overall = np.concatenate([y_pred_overall, y_pred_class])
        y_test_overall = np.concatenate([y_test_overall, y_test])
        k += 1

    print('CNN Classification Report: ')
    print (classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
    print('CNN Confusion Matrix: ')
    print (confusion_matrix(y_test_overall, y_pred_overall))
    print('CNN Classification Report: ')
    print(classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
    print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
    print('Macro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='macro')))
    print('Micro-averaged F1=', '%.3f' % (f1_score(y_test_overall, y_pred_overall, average='micro')))
    print('CNN Confusion Matrix: ')
    print(confusion_matrix(y_test_overall, y_pred_overall))


def CNN_train_test( X_train, y_train, X_test, y_test, classname ):
    print('X_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)
    n_samples, d, chan = X_train.shape
    num_classes = len(classname)
    Y_train = to_categorical(y_train, num_classes=num_classes)
    Y_test = to_categorical(y_test, num_classes=num_classes)

    model = _buildsimplemodel(d, chan, num_classes, kernel_size, filters )
    model.compile(loss=loss,optimizer=optimizer) #Must reset weights
    model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=1)
    y_pred = model.predict_classes(X_test, batch_size=1, verbose=2)
    # print 'CNN Classification Report: '
    print (classification_report(y_test, y_pred, target_names=classname, digits=3))
    # print 'Accuracy=', '%.2f %%' % (100*accuracy_score(y_test, y_pred))
    # print 'CNN Confusion Matrix: '
    print (confusion_matrix(y_test, y_pred))



if __name__ == '__main__':

    classname = ['class1', 'class2', 'class3']
    c = len(classname)   # classes

    # Generate dummy data
    n = 50 # samples
    chan = 2    # channels e.g. RGB in image
    d = 64 # signal features

    x_train = np.random.random((n, d, chan))
    y_train = np.random.randint(c, size=(n, 1)) # 1-D array
    #y_train = keras.utils.to_categorical(y_train, num_classes=c) # 1-out-of-c
    x_test = np.random.random((n, d, chan))
    y_test = np.random.randint(c, size=(n, 1))
    #y_test = keras.utils.to_categorical(y_test, num_classes=c)

    #print 'shape training patterns=', x_train.shape, '\nshape training labels=', y_train.shape
    #print x_train[0,0:5,0]  # first pattern, first five features
    #print y_train[0]        # first pattern, one-out-of-c label

    X = np.concatenate((x_train,x_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0)
    '''
    print 'shape X=', X.shape, '\nshape y=', y.shape
    print '\nx_train[0]=\n', x_train[0], '\ny_train[0]=\n', y_train[0]
    print '\nX[0]=\n', X[0], '\ny[0]=\n', y[0]
    '''
    kernel_size = 32    # size of the convolution filter (image would be e.g. tupel (3,3) )
    filters = 2 # number of convolution filters

    CNN( X, y, './', classname )
