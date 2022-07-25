import tensorflow as tf
from tensorflow import keras

import os
import glob
import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
from keras.losses import mse, binary_crossentropy

def readFiles(dir):
    # extract all .DAT files:
    files = glob.glob(dir + '/*.' + 'DAT')
    print('\nFound {0} files\n'.format(len(files)))

    # Read files and save into dictionary
    FIRST_FILE = True
    for file in files:
        with open(file) as f:
            lines = f.readlines()
            spectrum = [line.split()[1] for line in lines]
            spectrum= np.asarray(spectrum).astype(float)
            spectrum = np.expand_dims(spectrum, axis=0)
            try:
                X = np.vstack((X, spectrum))
            except(NameError):
                X = spectrum
            if FIRST_FILE:
                # read wavelength (only once, since constant)
                w_raw = [line.split()[0] for line in lines]
                w = np.asarray(w_raw).astype(float)
                FIRST_FILE = False
    return w, X

def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - \
                   np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    
    print('Mean for anomaly scores: ', np.mean(loss))
    
    return loss


pwd = os.getcwd()
dir = pwd + '\\..\\..\\04_Daten\\sample'
os.chdir(dir)

# extract spectras
_, X = readFiles(dir)
np.random.shuffle(X)
n_data = X.shape[0]
n_train = int(0.75*n_data)
dataX = X[:n_train]
testDataX = X[n_train:]

# Call neural network API
model = Sequential()
model.add(Dense(units=1024, activation='linear',input_dim=1024))
model.add(Dense(units=1024, activation='linear'))
model.add(Dense(units=1024, activation='linear'))
# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train the model
num_epochs = 1000
batch_size = 256
history = model.fit(x=dataX, y=dataX,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(dataX, dataX),
                    verbose=1)

predictions = model.predict(testDataX, verbose=1)
anomalyScoresAE = anomalyScores(testDataX, predictions)

print(anomalyScoresAE)

