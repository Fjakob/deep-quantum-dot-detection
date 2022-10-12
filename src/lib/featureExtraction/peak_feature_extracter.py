import numpy as np


class PeakFeatureExtracter():
    """ Perform feature extraction based on peak detection. """

    def __init__(self, peak_detector):
        self.detector = peak_detector
        self.features = list()

    def extract_from_spectrum(self, x):
        """ Extracts features from a single sample based on selected features. """
        v = x
        return v

    def extract_from_dataset(self, X):
        """ Extracts features from dataset based on selected features. """
        V = X
        return X

    def feature_selection(self, train_dataset):
        """ Selects most relevant features based on a training dataset. """
        X, Y = train_dataset

    