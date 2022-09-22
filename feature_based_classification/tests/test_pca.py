import numpy as np
from src.lib.dimensionReduction.pca import PCA

def test_instanciation():
    """ Tests the instanciation of the autoencoder class. """
    pca = PCA(latent_dim=12)
    assert isinstance(pca, PCA)


def test_dimensions():
    """ Tests latent and output dimension of PCA call. """
    pca = PCA(latent_dim=12)

    X_t = np.random.randn(400, 1024)
    pca.fit(X_t)

    X = np.random.randn(4,1024)
    Z, X_rec = pca.reduce(X, return_reconstruction=True)

    assert Z.shape[0] == 4 and Z.shape[1] == 12
    assert X.shape[0] == X_rec.shape[0] and X.shape[1] == X_rec.shape[1]


