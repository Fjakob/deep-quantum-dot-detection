import numpy as np
from scipy.optimize import minimize

from src.lib.baseClasses.peak_detector import PeakDetector

class ThresholdDetector(PeakDetector):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def detect(self, x):
        """ Return indices of peaks in signal x along with respective peak threshold. """
        n = len(x)
        peak_indices, thresholds = [], []
        for idx in range(n):
            if x[idx] > self.threshold:
                peak_indices.append(idx)
        
        if isinstance(peak_indices, list):
            # Merge consecutive peak indices that belong to the same peak
            peak_indices = self.merge_consecutive_peaks(peak_indices,x)
            n_peaks = len(peak_indices)
        elif not peak_indices:
            n_peaks = 0
        else:
            n_peaks = 1

        return peak_indices, n_peaks, self.threshold*np.ones(n)


    def optimize_parameters(self, labeled_data_set):
        """ Optimized N, T and N_protect in OS-CFAR-detector. """
        X, Y = labeled_data_set

        w_init = self.threshold
        initial_simplex = np.asarray([[w_init], [1.1*w_init]])

        minimize(self.cost_function, w_init, method='nelder-mead', args=(X, Y), 
                  options={'xatol': 1e-2, 'disp': False, 'initial_simplex': initial_simplex})
        
        print(f"Optimized Parameters: T={self.threshold}")


    def cost_function(self, w, X, Y):
        """ 
        Cost function for parameter optimization
        Constraint 1: N, T, N_protect > 0!
        Constraint 2: len(x) > N !
        """
        print(f"Parameters: T={w[0]}")

        # determine accuracy with updated parameters
        self.threshold = w[0]
        acc = self.accuracy(X, Y)
        print(f"Achieved accuracy: {acc}\n")
        return -acc