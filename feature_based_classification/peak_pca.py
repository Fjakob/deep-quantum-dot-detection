import numpy as np
import matplotlib.pyplot as plt
import pickle
from peakDetectors.OS_CFAR import OS_CFAR

def extract_principal_peaks():
    """For parameter tuning of OS-CFAR."""

    with open('dataSets/classificationData', 'rb') as f:
        dataSet = pickle.load(f)
    X = [x for _, x, _ in dataSet]

    detector = OS_CFAR(N=200, T=7, N_protect=20)
    peak_snippets = detector.isolate_peak_neighbourhoods(X, 15)

    X_peaks = np.asarray(peak_snippets)

    U, S, Vh = np.linalg.svd(X_peaks)

    principal_shapes = 10

    X_red = np.matmul(U[0:principal_shapes, :], X_peaks)

    for x in X_red:
        plt.plot(x)
        plt.show()


if __name__ == "__main__":
    extract_principal_peaks()