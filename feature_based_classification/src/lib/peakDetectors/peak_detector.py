import numpy as np
from itertools import groupby
from operator import itemgetter

class PeakDetector():
    """ Abstract peak detector class. """

    def detect(self, X):
        """ Return indices of peaks in signal x along with respective peak threshold. """
        raise NotImplementedError

    def merge_consecutive_peaks(self, peak_indices, x):
        """Merge multiple consecutive peak indices to the one corresponding to the highest value in x."""
        mergedPeaks = []
        for _, g in groupby(enumerate(peak_indices), lambda ix: ix[0]-ix[1]):
            # group is a sublist containing consecutive peak indices
            group = list(map(itemgetter(1), g))
            mergedPeaks.append(group[np.argmax(x[group])])
        return mergedPeaks

    def accuracy(self, X, Y):
        """ Returns detector accuracy on given dataset. """
        hit = 0
        for x, y in zip(X, Y):
            _, n_peak, _ = self.detect(x)
            hit += np.exp(np.log(0.5)*np.square(y-n_peak))
        accuracy = hit / len(X)
        return accuracy

    def optimize_parameters(self, labeled_data_set):
        raise NotImplementedError

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
