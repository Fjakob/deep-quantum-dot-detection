import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rnd
from scipy.signal import find_peaks, peak_widths
from itertools import groupby
from operator import itemgetter
import time


def OS_CFAR(x, N=32, T=5, k=None, N_protect=0):
    """Implementation of OS-CFAR Detector (see Literature)."""
    if k is None:
        k = round(3/4*N)
    else:
        assert k < N, "k cannot be bigger than N!"

    n = len(x)
    peaks, thresh = [], []
    for idx in range(n):
        # Collect Window around CUT
        # idx-N/2 | ... | idx-1 | (idx) | idx+1 | ... | idx+N/2
        X_left  = [x[idx+jdx] for jdx in range(int(N_protect/2)+1, int((N+N_protect)/2)+1) if idx+jdx < n-1]
        X_right = [x[idx-jdx] for jdx in range(int(N_protect/2)+1, int((N+N_protect)/2)+1) if idx-jdx > 0]
        # sort Window in ascending order
        X = X_left + X_right
        X.sort()
        # edit k, if Window length is not N (case e.g. in signal boundaries)
        if len(X) != N:
            k = round(3/4*len(X))
        # calculate threshold based on ordered statistics X
        threshold = X[k]*T
        thresh.append(threshold)
        # OS-CFAR test
        if x[idx] > threshold:
            peaks.append(idx)

    if isinstance(peaks, list):
        # Consecutive peak indices belong to the same peak: 
        peaks = mergeConsecutivePeaks(peaks,x)
        n_peaks = len(peaks)
    elif not peaks:
        # No peaks
        n_peaks = 0
    else:
        n_peaks = 1

    return peaks, n_peaks, np.asarray(thresh)


def mergeConsecutivePeaks(peak_indices, x):
    """Merge multiple consecutive peak indices to the one corresponding to the highest value in x."""
    mergedPeaks = []
    for _, g in groupby(enumerate(peak_indices), lambda ix: ix[0]-ix[1]):
        # group is a sublist containing consecutive peak indices
        group = list(map(itemgetter(1), g))
        mergedPeaks.append(group[np.argmax(x[group])])
    return mergedPeaks


if __name__ == '__main__':
    """For parameter tuning of OS-CFAR."""

    with open('dataSet', 'rb') as f:
        dataSet = pickle.load(f)
    rnd.shuffle(dataSet)

    #Choose from: 'Threshold', 'OS-CFAR'
    detection = 'OS-CFAR'
    
    hit=0
    t = time.time()
    for w, x, label in dataSet:
        y_peak = label[0]

        if detection=='Threshold':
            param_peak = 20.5
            idx_peak, _ = find_peaks(x, height=param_peak)
            n_peak = len(idx_peak)
            param_width = 0.9
            widths = peak_widths(x, idx_peak, rel_height=param_width)
        elif detection=='OS-CFAR':
            idx_peak, n_peak, thresh = OS_CFAR(x, N=200, T=7, N_protect=20)
            param_width = 0.7
            widths = peak_widths(x, idx_peak, rel_height=param_width)
        plt.plot(x)
        plt.plot(idx_peak, x[idx_peak], 'x')
        plt.plot(thresh)
        #plt.hlines(param_peak, 0, 1023, color='green')
        #plt.title("Peaks: {0}, f(x)={1}".format(int(y_peak), n_peak))
        plt.hlines(*widths[1:], color="C2")
        plt.grid()
        plt.show()
        hit += np.exp(np.log(0.5)*np.square(y_peak-n_peak))
        #programPause = input("Press the <ENTER> key to continue...")
    accuracy = hit / len(dataSet)

    print("Accuracy: {0}\%".format(accuracy*100))
    print("Elapsed time: {0:.2g}s".format(time.time()-t))

    # accuracy=0.5 == +-1 peak misclassified




