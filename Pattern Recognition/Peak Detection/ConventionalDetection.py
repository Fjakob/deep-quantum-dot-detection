import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as rnd
from scipy.signal import find_peaks, peak_widths

def CAGO_CFAR(x, N=32, T=5, N_protect=0):
    n = len(x)
    peaks = []
    n_peaks = 0
    thresh = []
    for idx in range(n):
        CUT = x[idx]
        # Collect Window around CUT
        X_left = []
        X_right = []
        # idx-N/2 | ... | idx-1 | (idx) | idx+1 | ... | idx+N/2
        for jdx in range(int((N+N_protect)/2)):
            if (jdx+1) > N_protect/2:
                if idx+jdx+1 < n-1:
                    X_right.append(x[idx+jdx+1])
            if (-jdx-1) < (-N_protect/2):
                if idx-jdx-1 > 0:
                    X_left.append(x[idx-jdx-1])
        Z = 2/N * np.max([np.asarray(X_left).sum(), np.asarray(X_right).sum()])
        thresh.append(Z*T)
        # OS-CFAR test
        if CUT > Z*T:
            peaks.append(idx) #peak
    if isinstance(peaks, list):
        peaks = mergeMountains(peaks,x)
        n_peaks = len(peaks)
    elif not peaks:
        n_peaks = 0
    else:
        n_peaks = 1
    return peaks, n_peaks, np.asarray(thresh)


def OS_CFAR(x, N=32, T=5, k=None, N_protect=0):
    # useful: N=32, k=27, T=10.7
    n = len(x)
    if k is None:
        k = round(3/4*N)
    peaks = []
    n_peaks = 0
    thresh = []
    for idx in range(n):
        CUT = x[idx]
        # Collect Window around CUT
        X = []
        # idx-N/2 | ... | idx-1 | (idx) | idx+1 | ... | idx+N/2
        for jdx in range(int((N+N_protect)/2)):
            if (jdx+1) > N_protect/2:
                if idx+jdx+1 < n-1:
                    X.append(x[idx+jdx+1])
            if (-jdx-1) < (-N_protect/2):
                if idx-jdx-1 > 0:
                    X.append(x[idx-jdx-1])
        # sort Window in ascending order
        X.sort()
        if len(X) != N:
            # case e.g. in signal boundaries
            k = round(3/4*len(X))
        Z = X[k]
        thresh.append(Z*T)
        # OS-CFAR test
        if CUT > Z*T:
            peaks.append(idx) #peak
    if isinstance(peaks, list):
        peaks = mergeMountains(peaks,x)
        n_peaks = len(peaks)
    elif not peaks:
        n_peaks = 0
    else:
        n_peaks = 1
    return peaks, n_peaks, np.asarray(thresh)


def mergeMountains(peaks, x):
    mountain=False
    peak_mem=[]
    out_peaks=[]
    for i, peak_idx in enumerate(peaks):
        try:
            if peak_idx+1 == peaks[i+1]:
                mountain=True
                if not peak_mem:
                    peak_mem.append(peak_idx)
                peak_mem.append(peaks[i+1])
            else:
                if not mountain:
                    out_peaks.append(peak_idx)
                else:
                    peak_idx = peak_mem[np.argmax(x[peak_mem])]
                    out_peaks.append(peak_idx)
                    mountain=False
                    peak_mem=[]
        except(IndexError):
            #last element
            if not mountain:
                    out_peaks.append(peak_idx)
            else:
                peak_idx = peak_mem[np.argmax(x[peak_mem])]
                out_peaks.append(peak_idx)
                mountain=False
                peak_mem=[]
    return out_peaks


if __name__ == '__main__':

    with open('dataSet', 'rb') as f:
        dataSet = pickle.load(f)

    #Choose from: 'Threshold', 'OS-CFAR', 'CAGO-CFAR'
    detection = 'OS-CFAR' 

    hit=0
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
            param_width = 0.9
            widths = peak_widths(x, idx_peak, rel_height=param_width)
        elif detection=='CAGO-CFAR':
            idx_peak, n_peak, thresh = CAGO_CFAR(x, N=256, T=7, N_protect=10)

        plt.plot(x)
        plt.plot(idx_peak, x[idx_peak], 'x')
        plt.plot(thresh)
        #plt.hlines(param_peak, 0, 1023, color='green')
        plt.title("Peaks: {0}, f(x)={1}".format(int(y_peak), n_peak))
        plt.hlines(*widths[1:], color="C2")
        plt.grid()
        plt.show()
        hit += np.exp(-0.69*np.square(y_peak-n_peak))
    print("Accuracy: {0}\%".format(hit/len(dataSet)*100))

    # accuracy=0.5 == +-1 peak misclassified


