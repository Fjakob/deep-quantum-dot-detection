import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.signal import peak_widths
from scipy.stats import entropy
from sklearn.model_selection import KFold

from src.lib.featureExtraction.norms import *


 
class FeatureExtracter():
    """ ... """

    def __init__(self, peak_detector, reconstructor, seed):
        np.random.seed(seed)
        self.detector = peak_detector
        self.reconstructor = reconstructor
        self.whiteNoise = np.random.normal(0.25, 2.52, size=1024)
        self.features = ["reconstruction_error"]

    

    def set_features(self, feature_list):
        """ ... """
        self.features = feature_list

    

    def extract_reconstruction_error(self, X):
        """ ... """
        if X.shape == 1:
            X = np.expand_dims(X, axis=0)

        X_norm, _, X_hat = self.reconstructor.normalize_and_extract(X)
        reconstruction_errors = window_loss(X_hat, X_norm)

        return np.expand_dims(reconstruction_errors, axis=1)



    def extract_peak_measures(self, X):
        """ Extracts peak related features. """
        features_X = []

        for x in tqdm(X):
            features_x = []

            ### Number of peaks, location of peaks:
            peak_idx, n_peak, _ = self.detector.detect(x)

            ### Peak related features
            if peak_idx:
                peaks = np.sort(x[peak_idx])

                # distance among peaks (if more than one existent):
                if len(peak_idx) > 1:
                    d_peaks = [abs(v - peak_idx[(i+1)%len(peak_idx)]) for i, v in enumerate(peak_idx)][:-1]
                    d_min = np.min(d_peaks)
                else:
                    d_min = 1024

                # Width among peaks:
                widths = peak_widths(x, peak_idx, rel_height=0.7)
                w_max = widths[0][np.argmax(x[peak_idx])]
                
                # Summed heights of non dominant peaks, normalized to maximum peak
                if len(peaks)>1:
                    min_to_max = np.mean(peaks[:-1]) / peaks[-1]
                else:
                    min_to_max = 0

                # Max value:
                x_max = np.max(x)
            else:
                # No peaks existent
                d_min = 0
                w_max = 0
                min_to_max = 1
                x_max = 0
            
            features_x.append(n_peak) if "n_peak" in self.features else None
            features_x.append(d_min) if "d_min" in self.features else None
            features_x.append(w_max) if "w_max" in self.features else None
            features_x.append(min_to_max) if "min_to_max" in self.features else None
            features_x.append(x_max) if "x_max" in self.features else None

            features_X.append(features_x)
        
        return np.asarray(features_X)

    

    def extract_noise_correlations(self, X):
        """ Calculates entropy with white noise"""
        
        entropies = list()
        for x in X:
            noise_normalized  = 0.5*(self.whiteNoise / np.max(np.abs(self.whiteNoise))) + 1
            signal_normalized = 0.5*(x / np.max(np.abs(x))) + 1
            entropies.append(entropy(noise_normalized, signal_normalized))

        return np.expand_dims(entropies, axis=1)



    def extract_from_dataset(self, X):
        """ Extracts features from dataset based on selected features. """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        print("Extracting features...")
        features = list()

        if "reconstruction_error" in self.features:
            error_features = self.extract_reconstruction_error(X)
            features.append(error_features)
            
        if [any for any in ["n_peak", "d_min", "w_max", "min_to_max", "x_max"] if any in self.features]:
            peak_features = self.extract_peak_measures(X)
            features.append(peak_features)

        if "noise_correlation" in self.features:
            noise_features = self.extract_noise_correlations(X)
            features.append(noise_features)

        return np.hstack(tuple(features))



    def evaluate_features(self, eval_dataset, regressor, folds):
        """ Mean R2-score of cross-validation on given data set with given regressor. """

        X, Y = eval_dataset
        V = self.extract_from_dataset(X)

        k_fold = KFold(n_splits=folds)
        r2_scores = list()

        print("Computing R2-scores...")
        for train_index, test_index in tqdm(k_fold.split(V)):
            V_train, V_test = V[train_index], V[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            regressor.fit(V_train, Y_train, verbose=False)
            r2_scores.append(regressor.score(V_test, Y_test))
        
        return np.mean(r2_scores)


    
    def feature_forward_selection(self, eval_dataset, regressor, folds=5):
        """ 
        Selects most relevant features based on forward selection algorithm: 

        1) Subsequentially add each feature to the feature set, compute accuracy metric
        2) Add feature with highest resulting accuary permanently to the set

        Regressor must provide a fit() and score() function
        As accuracy, mean R2-score of cross-validation on given data set with given regressor is used.
        """

        self.features = []

        features = ["reconstruction_error", 
                    "n_peak",
                    "d_min",
                    "w_max",
                    "min_to_max",
                    "x_max",
                    "noise_correlation"]

        iter = 1
        while len(features) > 1:

            print(f"-------------- ITERATION {iter} -------------- ")
            print(f"Feature list: \n{self.features}\n")
            significance_dict = dict()

            for feature in features.copy():
                print(f"Adding {feature}... ")
                self.features.append(feature)

                r2_performance = self.evaluate_features(eval_dataset, regressor, folds)
                print(f"Retrieved R2: {r2_performance}\n")
                significance_dict[feature] = r2_performance

                self.features.remove(feature)

            most_significant_feature = max(significance_dict, key=significance_dict.get)

            self.features.append(most_significant_feature)
            features.remove(most_significant_feature)

            iter += 1



    def feature_backward_elimination(self, eval_dataset, regressor, folds=5):
        """ 
        Selects most relevant features based on backward elimination algorithm: 

        1) Subsequentially remove each feature from the set, compute accuracy metric
        2) Remove feature with highest remaining accuary permanently from the set

        As accuracy, mean R2-score of cross-validation on given data set with given regressor is used.
        """

        self.features = ["reconstruction_error", 
                         "n_peak",
                         "d_min",
                         "w_max",
                         "min_to_max",
                         "x_max",
                         "noise_correlation"]

        r2_performance = self.evaluate_features(eval_dataset, regressor, folds)
        print(f"Reference initial R2: {r2_performance}\n")

        iter = 1
        while len(self.features) > 1:

            print(f"-------------- ITERATION {iter} -------------- ")
            print(f"Feature list: \n{self.features}\n")
            redundancy_dict = dict()
            features = self.features.copy()

            for feature in features:
                print(f"Removing {feature}... ")
                self.features.remove(feature)

                r2_performance = self.evaluate_features(eval_dataset, regressor, folds)
                redundancy_dict[feature] = r2_performance
                print(f"Remaining R2: {r2_performance}\n")

                self.features.append(feature)

            redundant_feature = max(redundancy_dict, key=redundancy_dict.get)
            self.features.remove(redundant_feature)

            print(f"Removed feature: {redundant_feature}\n")
            iter += 1


    
    def extract_latent_features(self, X):
        _, Z, _ = self.reconstructor.normalize_and_extract(X)
        return Z
