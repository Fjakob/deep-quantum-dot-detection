import numpy as np
from lib.peakDetectors.peak_detector import Peak_Detector

class OS_CFAR(Peak_Detector):
    """ Class of Ordered-Statistics Constant-False-Alarm-Rate (OS-CFAR) Detector. """
    def __init__(self, N=32, T=5, k=None, N_protect=0):
        self.N = N
        self.T = T
        self.k = k
        self.N_protect = N_protect


    def detect(self, x):
        """ TODO """
        N, T, k, N_protect = self.N, self.T, self.k, self.N_protect

        if k is None:
            k = round(3/4*N)
        else:
            assert k < N, "k cannot be bigger than N!"

        n = len(x)
        peak_indices, thresholds = [], []
        for idx in range(n):
            # Collect Window around CUT and sort values in ascending order
            # idx-N/2 | ... | idx-1 | (idx) | idx+1 | ... | idx+N/2
            X_left = [x[idx-jdx] for jdx in range(int(N_protect/2)+1, int((N+N_protect)/2)+1) if idx-jdx >= 0]
            X_right  = [x[idx+jdx] for jdx in range(int(N_protect/2)+1, int((N+N_protect)/2)+1) if idx+jdx <= n-1]
            X = X_left + X_right
            X.sort()

            # edit k, if Window length is not N (case e.g. in signal boundaries)
            if len(X) != N:
                k = int(np.floor(3/4*len(X)))

            threshold = X[k]*T
            thresholds.append(threshold)

            if x[idx] > threshold:
                peak_indices.append(idx)

        if isinstance(peak_indices, list):
            # Merge consecutive peak indices that belong to the same peak
            peak_indices = self.merge_consecutive_peaks(peak_indices,x)
            n_peaks = len(peak_indices)
        elif not peak_indices:
            n_peaks = 0
        else:
            n_peaks = 1

        return peak_indices, n_peaks, np.asarray(thresholds)


    def optimize_parameters(self, labeled_data_set):
        """ TODO """
        pass


    def isolate_peak_neighbourhoods(self, X, neighbourhood_width):
        """ Returns neighbourhoods of every detected peak in a data set. """
        peak_snippets = []
        for x in X:
            peak_indices, _, _ = self.detect(x)
            max_idx = len(x)

            # extract neighbourhood around every detected peak
            for peak_idx in peak_indices:
                idx_left = peak_idx - neighbourhood_width
                idx_right = peak_idx + neighbourhood_width + 1

                # Check if neighbourhood falls out of signal
                if idx_left < 0:
                    padding_left = np.zeros( abs(idx_left) )
                    x_snippet = np.concatenate( (padding_left, x[0:idx_right]) )
                elif idx_right > max_idx:
                    padding_right = np.zeros( idx_right - max_idx )
                    x_snippet = np.concatenate( (x[idx_left:max_idx], padding_right) )
                else:
                    x_snippet = x[idx_left:idx_right]

                peak_snippets.append(x_snippet)

        return peak_snippets


