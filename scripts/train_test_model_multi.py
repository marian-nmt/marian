#!/usr/bin/env python

import sys
import os
import numpy as np
import time
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


def baseline_model(pixels_count, classes_count):
    model = Sequential()
    model.add(Dense(2000, input_dim=pixels_count, init='normal', activation='tanh'))
    model.add(Dense(2000, init='normal', activation='tanh'))
    model.add(Dense(2000, init='normal', activation='tanh'))
    model.add(Dense(2000, init='normal', activation='tanh'))
    model.add(Dense(2000, init='normal', activation='tanh'))
    model.add(Dense(classes_count, input_dim=100, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

    X_train = X_train / 255
    X_test = X_test / 255

    ### Change classes to one hot encoding matrixes

    y_train = np_utils.to_categorical(y_train)
    classes_count = y_train.shape[1]
    print "Y shape: ", y_train.shape
    
    y_test = np_utils.to_categorical(y_test)

    # Train weight matrix

    # Build the model
    model = baseline_model(pixels_count, classes_count)
    # Fit the model
    
    start = time.time();
    model.fit(X_train, y_train, nb_epoch=10, batch_size=200, verbose=2)
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
