#!/usr/bin/env python

import sys
import os
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


def baseline_model(pixels_count, classes_count):
    model = Sequential()
    # model.add(Dense(pixels_count, input_dim=pixels_count, init='normal', activation='relu'))
    model.add(Dense(classes_count, input_dim=pixels_count, init='normal', activation='softmax'))
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
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    ### Weight and bias matrixes - we extract them from the model
    
    # weights_ones = np.ones((pixels_count, classes_count))
    # print weights_ones.shape

    weights, bias = model.get_weights()
    print weights.shape
    print bias.shape
    print bias

    ### We calculate lr using softmax!

    dot_out = np.dot(X_train, weights)
    print "dot_out shape: ", dot_out.shape
    # print dot_out[:10]
    
    add_out = np.add(bias, dot_out)
    print "add_out shape: ", add_out.shape
    # print add_out[:10]
    
    # lr = np.around(softmax(add_out), decimals = 6)
    lr = softmax(add_out)
    print "lr shape: ", lr.shape
    # print lr[:10]
    # print np.count_nonzero(lr)i

    ### Save model to npz files
    if not os.path.exists("test_model_single"):
        os.makedirs("test_model_single")
    np.savez("test_model_single/model", weights = weights, bias = bias)

    print "Model saved! Check test_model_single directory"
