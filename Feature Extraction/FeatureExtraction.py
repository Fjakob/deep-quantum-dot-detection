import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rnd
from scipy.signal import peak_widths
from scipy.stats import entropy

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
    # Entropy with white noise:
    np.random.seed(1)
    whiteNoise = np.random.normal(0.25, 2.52, size=1024)
    noiseCorr = entropy(0.5*(whiteNoise/np.max(np.abs(whiteNoise)))+1, 0.5*(x/np.max(np.abs(x)))+1)
    #features["White Noise Entropy"] = noiseCorr
    if peak_idx:
        peaks = np.sort(x[peak_idx])
        # Width and distance among peaks:
        widths = peak_widths(x, peak_idx, rel_height=0.7)
        d_peaks = [abs(v - peak_idx[(i+1)%len(peak_idx)]) for i, v in enumerate(peak_idx)][:-1]
        # Minimum distance:
        if d_peaks:
            d_min = np.min(d_peaks)
        else:
            d_min = 1024
        features["d_min"] = d_min
        # Width of maximum peak:
        w_maxPeak = widths[0][np.argmax(x[peak_idx])]
        features["w_maxPeak"] = w_maxPeak
        # Summed heights of non dominant peaks, normalized to maximum peak
        if len(peaks)>1:
            minorToMajor = np.mean(peaks[:-1]) / peaks[-1]
        else:
            minorToMajor = 0
        #features["Minor To Major Peaks"] = minorToMajor
        # SNR or max value:
        x_max = peaks[-1]
        #features["Max Peak height"] = x_max
    else:
        pass
        # No peaks existent
        features["d_min"] = 0
        features["w_maxPeak"] = 1024
        #features["Minor To Major Peaks"] = 2
        #features["Max Peak height"] = 0
    return features


if __name__ == '__main__':

    # load data set
    X, Y = loadDataSet('dataSet')

    spec = X[8]
    print(extractFeatures(spec))
    print(np.asarray(list(extractFeatures(spec).values())))

    plt.plot(spec)
    plt.show()

