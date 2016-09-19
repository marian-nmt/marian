#!/usr/bin/env python

import sys
import os
import numpy as np
import time
import theano

np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
np.random.seed(42)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from keras.optimizers import Adam, SGD

#class Adam2(SGD):
#    def get_gradients(self, loss, params):
#       print "lalala"
#       grads = K.gradients(loss, params)
#       if hasattr(self, 'clipnorm') and self.clipnorm > 0:
#           norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
#           grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
#       if hasattr(self, 'clipvalue') and self.clipvalue > 0:
#           grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
#       grads = [theano.printing.Print('Gradient')(g) for g in grads]
#       return grads
#
#
#X = 123456789
#Y = 362436069
#Z = 521288629
#W = 88675123
#
#def xorshift():
#    global X, Y, Z, W
#    t = (X ^ (X << 11)) % 1000
#    X = Y
#    Y = Z
#    Z = W
#    W = (W ^ (W >> 19) ^ t ^ (t >> 8)) % 1000
#    return 0.1 * ((W % 1000)/1000.0) - 0.05

#def xorshift_init(shape, name=None):
#    init = np.array([xorshift() for i in range(shape[0] * shape[1])]).reshape(shape)
#    return K.variable(init, name=name)

def baseline_model(pixels_count, classes_count):
    model = Sequential()
#    model.add(Dropout(0.2, input_shape=(pixels_count,)))
    model.add(Dense(2048, input_dim=pixels_count, init='uniform', activation='relu'))
#    model.add(Dense(2048, init='uniform', activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(2048, init='uniform', activation='relu'))
    model.add(Dense(2048, init='uniform', activation='relu'))
    model.add(Dense(2048, init='uniform', activation='relu'))
    model.add(Dense(2048, init='uniform', activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(classes_count, init='uniform', activation='softmax'))

    opt = Adam(lr=0.0002);
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    ### Load trainset from mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    ### Flatten pictures into vectors
   
    pixels_count = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], pixels_count).astype('float32')
    print "X shape: ", X_train.shape

    X_test = X_test.reshape(X_test.shape[0], pixels_count).astype('float32')

    ### Normalize data to (0, 1)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    ### Change classes to one hot encoding matrixes

    y_train = np_utils.to_categorical(y_train)
    classes_count = y_train.shape[1]
    print "Y shape: ", y_train.shape
    
    y_test = np_utils.to_categorical(y_test)

    # Train weight matrix

    # Build the model
    model = baseline_model(pixels_count, classes_count)
    
    #for layer in model.layers:
    #    print layer.get_weights() 
    # Fit the model
    
    start = time.time();
    model.fit(X_train, y_train, nb_epoch=10, batch_size=200, verbose=2, shuffle=True)

    print "Time elapsed", time.time() - start, "s"
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    ### Weight and bias matrixes - we extract them from the model
    
    # weights_ones = np.ones((pixels_count, classes_count))
    # print weights_ones.shape

    #weights1, bias1, weights2, bias2 = model.get_weights()
    ### Save model to npz files
    #if not os.path.exists("test_model_multi"):
    #    os.makedirs("test_model_multi")
    # np.savez("test_model_multi/model", *model)
    #np.savez("test_model_multi/model", weights1 = weights1, bias1 = bias1, weights2 = weights2, bias2 = bias2)

    #print "Model saved! Check test_model_multi directory"
