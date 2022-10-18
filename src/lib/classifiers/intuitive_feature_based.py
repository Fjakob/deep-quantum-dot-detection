import numpy as np


class PeakFeatureBasedRater():
    """ Class for spectrum rating """
    def __init__(self, feature_extracter, scaler, regressor):
        self.feature_extracter  = feature_extracter
        self.scale, self.mean = scaler
        self.regressor = regressor

    def rate(self, X):
        V = self.feature_extracter.extract_from_dataset(X)
        V = (V - self.mean) / self.scale
        if len(V.shape) == 1:
            V = np.expand_dims(V, axis=0)
        y = self.regressor.predict(V)
        return y