import numpy as np
import matplotlib.pyplot as plt
import os

import random as rnd
import glob
from scipy.signal import find_peaks, peak_widths

def readFile(dir):
    # extract all .DAT files:
    files = glob.glob(dir + '/*.' + 'DAT')
    print('\nFound {0} files\n'.format(len(files)))

    # Read files and save into dictionary 
    FIRST_FILE = True
    file = files[0]
    with open(file) as f:
        lines = f.readlines()
        spectrum = [line.split()[1] for line in lines]
        spectrum= np.asarray(spectrum).astype(float)
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


def AvgWin(x, M):
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
dir = pwd + '\\..\\..\\..\\04_Daten\\Maps_for_ISYS\\2021-09-17_map04_3000um_3000um'
os.chdir(dir)

# Prepare Data Set
w, x = readFile(dir)
#plt.plot(w, x)
#plt.show()


peaks, _ = find_peaks(x, height=10)
print(peaks)
results_half = peak_widths(x, peaks, rel_height=0.5)
results_half[0] 

plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.hlines(*results_half[1:], color="C2")
#plt.hlines(*results_full[1:], color="C3")
plt.show()