import numpy as np
import torch

class SpectrumClassifier():
    """ Class for spectrum rating """

    def __init__(self, reconstructer, regressor):
        self.reconstructer  = reconstructer
        self.regressor = regressor

    def rate(self, X):
        pass
