import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random as rnd

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


def toSpectogram(x, n_window):
    for idx in range(len(x) - n_window):
        x_windowed = x[idx:idx+n_window]
        np.expand_dims(x_windowed, axis=0)
        try:
            X = np.vstack((X, x_windowed))
        except(NameError):
            # first iteration
            X = x_windowed
    return np.transpose(X)


if __name__ == '__main__':

    # set directories
    pwd = os.getcwd()
    dir = pwd + '\\..\\..\\04_Daten\\Maps_for_ISYS\\2022-07-11_map18_3110um_0700um_100x50_650mV-1250mV'
    os.chdir(dir)

    # extract spectras
    wavelengths, spectras = readFiles(dir)


    X = toSpectogram(spectras[300], 512)

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.plot(wavelengths, spectras[0])
    ax = fig.add_subplot(1,2,2)
    ax.imshow(X, cmap='bwr', interpolation='none')
    plt.show()

    fig.savefig('Spectogram.pdf',format='pdf')