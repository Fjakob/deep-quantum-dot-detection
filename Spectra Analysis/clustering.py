import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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



if __name__ == '__main__':

    # set directories
    pwd = os.getcwd()
    dir = pwd + '\\..\\..\\04_Daten\\sample'
    os.chdir(dir)

    # extract spectras
    wavelengths, spectras = readFiles(dir)

    mean_X = 0#np.mean(spectras, axis=0)
    var_X  = 1#np.var(spectras, axis=0)
    X = (spectras - mean_X) / var_X
    pca = PCA(n_components=10)
    X = pca.fit_transform(np.transpose(X))

    for idx in range(10):
        plt.figure()
        eigenSpectrum = X[:,idx]*var_X + mean_X
        plt.plot(wavelengths, eigenSpectrum)
        plt.show()

