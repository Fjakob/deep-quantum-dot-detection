import numpy as np
from tqdm import tqdm
from scipy.signal import peak_widths
from scipy.stats import entropy
from sklearn.model_selection import KFold


class PeakFeatureExtracter():
    """ Perform feature extraction based on peak detection. """

    def __init__(self, peak_detector, seed):
        np.random.seed(seed)
        self.detector = peak_detector
        self.whiteNoise = np.random.normal(0.25, 2.52, size=1024)
        #self.features = ["n_peak", "d_min", "w_maxPeak", "minorToMajor", "x_max", "noise_correlation"]
        self.features = ["x_max", "noise_correlation"]


    def extract_from_spectrum(self, x):
        """ Extracts features from a single sample based on selected features. """
        features = []

        ### Number of peaks, location of peaks:
        peak_idx, n_peak, _ = self.detector.detect(x)
        features.append(n_peak) if "n_peak" in self.features else None

        ### Peak related features
        if peak_idx:
            peaks = np.sort(x[peak_idx])

            # distance among peaks (if more than one existent):
            if len(peak_idx) >= 2:
                d_peaks = [abs(v - peak_idx[(i+1)%len(peak_idx)]) for i, v in enumerate(peak_idx)][:-1]
                d_min = np.min(d_peaks)
            else:
                d_min = 1024

            # Width among peaks:
            widths = peak_widths(x, peak_idx, rel_height=0.7)
            w_maxPeak = widths[0][np.argmax(x[peak_idx])]
            
            # Summed heights of non dominant peaks, normalized to maximum peak
            if len(peaks)>1:
                minorToMajor = np.mean(peaks[:-1]) / peaks[-1]
            else:
                minorToMajor = 0

            # Max value:
            x_max = np.max(x)
        else:
            # No peaks existent
            d_min = 0
            w_maxPeak = 0
            minorToMajor = 1
            x_max = 0
        
        features.append(d_min) if "d_min" in self.features else None
        features.append(w_maxPeak) if "w_maxPeak" in self.features else None
        features.append(minorToMajor) if "minorToMajor" in self.features else None
        features.append(x_max) if "x_max" in self.features else None

        ### Entropy with white noise (parameters estimated from noise):
        noise_normalized  = 0.5*(self.whiteNoise/np.max(np.abs(self.whiteNoise)))+1
        signal_normalized = 0.5*(x/np.max(np.abs(x)))+1
        noise_correlation = entropy(noise_normalized, signal_normalized)*10
        features.append(noise_correlation) if "noise_correlation" in self.features else None

        return features


    def extract_from_dataset(self, X):
        """ Extracts features from dataset based on selected features. """
        if len(X.shape) == 1:
            V = self.extract_from_spectrum(X)
        else:
            V = []
            print("Extracting features...")
            for x in tqdm(X):
                V.append(self.extract_from_spectrum(x))
        return np.asarray(V)


    def feature_selection(self, regressor, eval_dataset):
        """ Selects most relevant features based on a training dataset. """
        X, Y = eval_dataset

        k_fold = KFold(n_splits=5)

        R2 = []
        V = self.extract_from_dataset(X)
        for train_index, test_index in k_fold.split(V):
            V_train, V_test = V[train_index], V[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            gpr = regressor.fit(V_train, Y_train)
            fit = gpr.score(V_test, Y_test)
            R2.append(fit)
        print(f"Reference initial R2: {np.mean(R2)}\n")

        iter = 1
        while len(self.features) > 1:
            print(f"-------------- ITERATION {iter} -------------- ")
            redundancy_dict = dict()
            features = self.features.copy()
            for feature in features:
                print(f"Removing {feature}")
                self.features.remove(feature)
                V = self.extract_from_dataset(X)
                R2 = []
                for train_index, test_index in k_fold.split(V):
                    V_train, V_test = V[train_index], V[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    gpr = regressor.fit(V_train, Y_train)
                    fit = gpr.score(V_test, Y_test)
                    R2.append(fit)
                print(f"R2 after removing: {np.mean(R2)}\n")
                redundancy_dict[feature] = np.mean(R2)
                self.features.append(feature)
                print(self.features)
            print(redundancy_dict)
            redundant_feature = max(redundancy_dict, key=redundancy_dict.get)
            self.features.remove(redundant_feature)
            print(f"Removed feature: {redundant_feature}\n")
            iter += 1

