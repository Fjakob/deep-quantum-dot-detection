import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random as rnd

import os
import glob
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

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

def anomalyScores(X_true, X_reconstructed):
    loss = np.sum((np.array(X_true) - \
                   np.array(X_reconstructed))**2, axis=1)
    loss = pd.Series(data=loss)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))    
    return loss


pwd = os.getcwd()
dir = pwd + '\\..\\..\\04_Daten\\Maps_for_ISYS\\2021-09-17_map02_3000um_1800um'
os.chdir(dir)

# Prepare Data Set
_, X = readFiles(dir)
np.random.shuffle(X)

n_data = X.shape[0]
n_train = int(0.6*n_data)
n_val   = int(0.2*n_data)

dataX = X[:n_train]
valDataX = X[n_train:n_train+n_val]
testDataX = X[n_train+n_val:]

# Call neural network API
model = Sequential()
model.add(Dense(units=1024, activation='linear',input_dim=1024))
#model.add(Dense(units=1024, activation='linear'))
model.add(Dense(units=1024, activation='linear'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train the model
num_epochs = 200
batch_size = 32
history = model.fit(x=dataX, y=dataX,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(valDataX, valDataX),
                    verbose=1)

predictions = model.predict(testDataX, verbose=1)
anomalyScoresAE = anomalyScores(testDataX, predictions)

anomalyThreshold = 0.2

noiseIdx = anomalyScoresAE[anomalyScoresAE <= anomalyThreshold].index.values
qDotsIdx = anomalyScoresAE[anomalyScoresAE > anomalyThreshold].index.values

nNoise = len(noiseIdx)
nQDots = len(qDotsIdx)
nmbSamples = 5

print('\nFound {0} spectras and {1} noisy\n'.format(nQDots, nNoise))

fig = plt.figure()
for idx in range(nmbSamples):
    r1 = rnd.randint(0, nNoise-1)
    r2 = rnd.randint(0, nQDots-1)
    ax = fig.add_subplot(nmbSamples,2,2*idx+1)
    ax.plot(testDataX[noiseIdx[r1]])
    ax = fig.add_subplot(nmbSamples,2,2*idx+2)
    ax.plot(testDataX[qDotsIdx[r2]])
plt.show()
