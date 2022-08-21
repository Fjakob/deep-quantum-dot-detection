import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rnd
from scipy.signal import peak_widths

from PeakDetection import OS_CFAR


def loadDataSet(filename):
    with open(filename, 'rb') as f:
        dataSet = pickle.load(f)
    for _, x, label in dataSet:
        y = label[1]
        try:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
        except(NameError):
            X = x
            Y = y
    return X, Y


def extractFeatures(x):
    features = dict()
    # Number of peaks, location of peaks:
    peak_idx, n_peak, _ = OS_CFAR(x, N=200, T=7, N_protect=20)
    features["N_peaks"] = n_peak
    # Width of peaks:
    widths = peak_widths(x, peak_idx, rel_height=0.7)
    features["w_peaks"] = widths[0] 
    # Distance among peaks:
    d_peaks = [abs(v - peak_idx[(i+1)%len(peak_idx)]) for i, v in enumerate(peak_idx)][:-1]
    features["d_peaks"] = d_peaks
    # Minimum distance:

    # Width of maximum peak:

    # Average widths: (might be highly correlated!)

    # SNR or max value:

    # Correlation with white noise:

    return features


if __name__ == '__main__':

    # load data set
    X, Y = loadDataSet('dataSet')

    print(extractFeatures(X[2]))

    plt.plot(X[2])
    plt.show()

