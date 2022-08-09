import numpy as np
import matplotlib.pyplot as plt
import os

import random as rnd
import glob

def readFiles(dir):
    # extract all .DAT files:
    files = glob.glob(dir + '/*.' + 'DAT')
    print('\nFound {0} files\n'.format(len(files)))

    # Read files and save into dictionary 
    FIRST_FILE = True
    idx=0
    for file in files:
        print(idx)
        idx+=1
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


def thresholdDetector(x, M):
    n = len(x)
    G = []
    for idx in range(n):
        weightedSum = x[idx]
        for jdx in range((M-1)/2):
            if idx+jdx+1 < n-1:
                weightedSum += x[idx+jdx+1]
            if idx-jdx-1 > 0:
                weightedSum += x[idx-jdx-1]
        G.append(weightedSum)
    return np.asarray(G)



pwd = os.getcwd()
dir = pwd + '\\..\\..\\..\\04_Daten\\Maps_for_ISYS\\2021-09-17_map06_3000um_0650um'
os.chdir(dir)

# Prepare Data Set
_, X = readFiles(dir)
plt.plot(X[0,:])
plt.show()
