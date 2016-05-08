import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
import time

inputs = Input(shape=(2,))
x = Dense(5000, activation='sigmoid')(inputs)
x = Dense(5000, activation='sigmoid')(x)
x = Dense(5000, activation='sigmoid')(x)
predictions = Dense(1, activation='sigmoid')(x)

X = np.array([
 0, 0,
 0, 1,
 1, 0,
 1, 1]).reshape((4,2))

Y = np.array([0, 1, 1, 0]).reshape((4,1))

#sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

start = time.time()
for i in range(10):
    model.fit(X, Y, nb_epoch=200, verbose=0)
    print model.predict(X)
    print model.evaluate(X, Y, verbose=0)
end = time.time()
print(end - start)


