
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



pwd = os.getcwd()
dir = pwd + '\\..\\..\\04_Daten\\sample'
os.chdir(dir)

# Prepare Data Set
_, X = readFiles(dir)
