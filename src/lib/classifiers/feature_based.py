import numpy as np
import torch

class SpectrumClassifier():
    """ Class for spectrum rating """

    def __init__(self, feature_extracter, regressor):
        self.feature_extracter  = feature_extracter
        self.regressor          = regressor

    def rate(self, X):
        pass

    def score(self, X):
        pass