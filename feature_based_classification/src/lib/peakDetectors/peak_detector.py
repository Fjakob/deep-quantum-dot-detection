import numpy as np
from itertools import groupby
from operator import itemgetter

class Peak_Detector():
    """ Abstract peak detector class. """

    def detect():
        raise NotImplementedError

    def merge_consecutive_peaks(self, peak_indices, x):
        """Merge multiple consecutive peak indices to the one corresponding to the highest value in x."""
        mergedPeaks = []
        for _, g in groupby(enumerate(peak_indices), lambda ix: ix[0]-ix[1]):
            # group is a sublist containing consecutive peak indices
            group = list(map(itemgetter(1), g))
            mergedPeaks.append(group[np.argmax(x[group])])
        return mergedPeaks

    def optimize_parameters(self, labeled_data_set):
        raise NotImplementedError

    def isolatePeakNeighbourhood(dataSet, neighbourhood_width):
        raise NotImplementedError
