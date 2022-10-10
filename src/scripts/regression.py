from __config__ import *

from tqdm import tqdm
from scipy.signal import peak_widths
from scipy.stats import entropy
from lib.peakDetectors.os_cfar import OS_CFAR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.model_selection import KFold


def loadDataSet(filename):
    with open(filename, 'rb') as f:
        dataSet = pickle.load(f)
    for _, x, y in dataSet:
        try:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
        except(NameError):
            X = x
            Y = y
    return X, Y

def extractFeatures(x, detector):
    features = dict()
    # Number of peaks, location of peaks:
    peak_idx, n_peak, _ = detector.detect(x)
    features["N_peaks"] = n_peak
    # Entropy with white noise:
    np.random.seed(1)
    whiteNoise = np.random.normal(0.25, 2.52, size=1024)
    noiseCorr = entropy(0.5*(whiteNoise/np.max(np.abs(whiteNoise)))+1, 0.5*(x/np.max(np.abs(x)))+1)
    features["White Noise Entropy"] = noiseCorr
    if peak_idx:
        peaks = np.sort(x[peak_idx])
        # Width and distance among peaks:
        widths = peak_widths(x, peak_idx, rel_height=0.7)
        d_peaks = [abs(v - peak_idx[(i+1)%len(peak_idx)]) for i, v in enumerate(peak_idx)][:-1]
        # Minimum distance:
        if d_peaks:
            d_min = np.min(d_peaks)
        else:
            d_min = 1024
        features["d_min"] = d_min
        # Width of maximum peak:
        w_maxPeak = widths[0][np.argmax(x[peak_idx])]
        features["w_maxPeak"] = w_maxPeak
        # Summed heights of non dominant peaks, normalized to maximum peak
        if len(peaks)>1:
            minorToMajor = np.mean(peaks[:-1]) / peaks[-1]
        else:
            minorToMajor = 0
        features["Minor To Major Peaks"] = minorToMajor
        # SNR or max value:
        x_max = peaks[-1]
        features["Max Peak height"] = x_max
    else:
        # No peaks existent
        features["d_min"] = 0
        features["w_maxPeak"] = 1024
        features["Minor To Major Peaks"] = 2
        features["Max Peak height"] = 0
    return features

def to_features(X, detector):
        for x in X:
            v = np.asarray(list(extractFeatures(x, detector).values()))
            try:
                V = np.vstack((V, v))
            except(NameError):
                V = v
        return V


if __name__ == "__main__":

    # Data loading
    X, Y = loadDataSet('dataSets/regressionData')
    X, Y = shuffle(X, Y)
    Y = np.ravel(Y) #delete inner dimension

    detector = OS_CFAR(N=200, T=7, N_protect=20)
    V = to_features(X, detector)

    # (If data set is big:) Splitting Data into Train and Test
    V_train, V_test, Y_train, Y_test = train_test_split(V, Y, test_size=0.15)

    print("Retrieved {} training and {} test data points".format(len(Y_train),len(Y_test)))

    # (Optional:) Data scaling

    scaler = StandardScaler()
    scaler.fit(V)
    scale, mean = scaler.scale_, scaler.mean_

    V_sc = scaler.transform(V)
    #V_train, V_test, Y_train, Y_test = train_test_split(V_sc, Y, test_size=0.3)

    # Gaussian Process Regression
    kernel_list = {"Linear": kernels.DotProduct() + kernels.WhiteKernel(), 
                "Gaussian" :kernels.RBF(),
                "Matérn": kernels.Matern(nu=1/2),
                "Rational quadratic": kernels.RationalQuadratic()}

    for name, kernel in kernel_list.items():
        k_fold = KFold(n_splits=5)
        R2 = []
        idx = 1
        for train_index, test_index in k_fold.split(V):
            V_train, V_test = V[train_index], V[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            gpr = GaussianProcessRegressor(kernel=kernel).fit(V_train, Y_train)
            fit = gpr.score(V_test, Y_test)
            print("R2 score on fold {}: {:.2f}".format(idx, fit))
            R2.append(fit)
            idx += 1

    print("Mean of R2 scores for " + name +  " kernel: {:.2f}".format(np.mean(R2)))

    gpr = GaussianProcessRegressor(kernel=kernel_list["Linear"]).fit(V, Y)
    print("Training R2-score: {}".format(gpr.score(V,Y)))

    for idx in range(10):
        x_test, y_test = X[idx], Y[idx]
        plt.plot(x_test)
        plt.show()
        y_bar, var = gpr.predict(scaler.transform([to_features([x_test], detector)]), return_std=True)
        print("Real label: {} \nPredicted: {} (with variance: {})".format(y_test, y_bar[0], var[0]))

