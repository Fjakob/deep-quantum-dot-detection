import numpy as np
from scipy.optimize import minimize

from src.lib.baseClasses.peak_detector import PeakDetector


class OS_CFAR(PeakDetector):
    """ Class of Ordered-Statistics Constant-False-Alarm-Rate (OS-CFAR) Detector. """
    def __init__(self, N=32, T=5, k=None, N_protect=0):

        assert 0 < N, "N must be bigger than zero!"
        assert 0 <= N_protect, "N_protect must be bigger than or equal to zero!" 
        if k is None:
            k = round(3/4*N)
        else:
            assert 0 < k, "k must be bigger than zero!"
            assert k < N, "k must be smaller than N!"        

        self.N = N
        self.T = T
        self.k = k
        self.N_protect = N_protect
        self.opt_hist = []


    def detect(self, x):
        """ Return indices of peaks in signal x along with respective peak threshold. """
        N, T, k, N_protect = self.N, self.T, self.k, self.N_protect

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
        """ Optimized N, T and N_protect in OS-CFAR-detector. """
        X, Y = labeled_data_set

        w_init = np.asarray([self.N, self.T, self.N_protect])
        initial_simplex = np.asarray([w_init,
                                     [32, 5, 1],
                                     [100, 6, 60],
                                     [300, 4, 10]])

        self.opt_hist = []

        minimize(self.cost_function, w_init, method='nelder-mead', args=(X, Y), 
                  options={'maxiter': 20, 'disp': False, 'initial_simplex': initial_simplex})
        
        print(f"Optimized Parameters: N={self.N}, T={self.T}, N_protect={self.N_protect}")


    def cost_function(self, w, X, Y):
        """ 
        Cost function for parameter optimization
        Constraint 1: N, T, N_protect > 0 !
        Constraint 2: len(x) > N !
        """
        N, T, N_protect = int(round(w[0])), float(w[1]), int(round(w[2]))
        print(f"Parameters: N={N}, T={T}, N_protect={N_protect}")

        # check constraint violation
        if (w <= 0).any() or w[0] >= len(X[0]):
            print("Invalid parameters, restart.\n")
            return 30.0

        # determine accuracy with updated parameters
        self.N, self.T, self.N_protect = N, T, N_protect
        acc = self.accuracy(X, Y) 
        self.opt_hist.append([acc, N, T, N_protect])
        print(f"Achieved accuracy: {acc}\n")
        return acc + N/1024


