import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse.linalg as linalg
from lib.peakDetectors.OS_CFAR import OS_CFAR
from lib.dimensionReduction.pca import PCA


def extract_principal_peaks():
    """ PCA analysis of peak snippets. """

    # load the dataset
    with open('dataSets/classificationData', 'rb') as f:
        dataSet = pickle.load(f)
    X = [x for _, x, _ in dataSet]

    # extract every peak shape from the dataset
    detector = OS_CFAR(N=200, T=7, N_protect=20)
    isolated_peaks = detector.isolate_peak_neighbourhoods(X, neighbourhood_width=20)

    # create and fit principal component analyzer
    X_train = np.asarray(isolated_peaks)
    pca = PCA(latent_dim = 10)
    pca.fit(X_train)
    pca.plot_reconstructions(X_train, nbr_plots=15)


if __name__ == "__main__":
    extract_principal_peaks()
