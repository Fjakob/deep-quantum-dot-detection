import numpy as np
import torch

class SpectrumClassifier():
    """ Class for spectrum rating """

    def __init__(self, feature_extracter, regressor):
        self.feature_extracter  = feature_extracter
        self.regressor          = regressor

    def rate(self, X):
        V = self.feature_extracter.extract_from_dataset(X, rescale=False, verbose=False)
        Y = self.regressor.predict(V)
        return Y

    def score(self, X, Y):
        V = self.feature_extracter.extract_from_dataset(X, rescale=False, verbose=False)
        return self.regressor.score(V, Y)