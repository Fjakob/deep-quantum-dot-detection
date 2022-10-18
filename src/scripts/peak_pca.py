from config.imports import *

from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.peakDetectors.threshold_detector import ThresholdDetector
from src.lib.featureExtraction.pca import PCA


def extract_principal_peaks():
    """ PCA analysis of peak snippets. """

    # load the dataset
    with open('dataSets/classificationData', 'rb') as f:
        dataSet = pickle.load(f)
    X = [x for _, x, _ in dataSet[:100]]
    Y = [y[0] for _, _, y in dataSet[:100]]

    # extract every peak shape from the dataset
    detector = OS_CFAR(N=190, T=6.9, N_protect=20)
    #detector = ThresholdDetector()
    #detector.optimize_parameters((X,Y))
    isolated_peaks = detector.isolate_peak_neighbourhoods(X, neighbourhood_width=20)

    # create and fit principal component analyzer
    X_train = np.asarray(isolated_peaks)
    pca = PCA(latent_dim = 10)
    pca.fit(X_train)
    pca.plot_reconstructions(X_train, nbr_plots=15)


if __name__ == "__main__":
    extract_principal_peaks()


