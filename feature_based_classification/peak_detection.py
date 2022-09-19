import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, peak_widths
import time
from peakDetectors.OS_CFAR import OS_CFAR

def detection():
    """For parameter tuning of OS-CFAR."""

    with open('dataSets/classificationData', 'rb') as f:
        dataSet = pickle.load(f)

    hit=0
    t = time.time()

    detector = OS_CFAR(N=200, T=7, N_protect=20)

    for w, x, label in dataSet:
        y_peak = label[0]

        if detection=='Threshold':
            param_peak = 20.5
            idx_peak, _ = find_peaks(x, height=param_peak)
            n_peak = len(idx_peak)
            param_width = 0.9
            widths = peak_widths(x, idx_peak, rel_height=param_width)
        elif detection=='OS-CFAR':
            idx_peak, n_peak, thresholds = detector.detect(x)
            param_width = 0.7
            widths = peak_widths(x, idx_peak, rel_height=param_width)
        plt.plot(x)
        plt.plot(idx_peak, x[idx_peak], 'x')
        plt.plot(thresholds)
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


if __name__ == '__main__':
    detection()