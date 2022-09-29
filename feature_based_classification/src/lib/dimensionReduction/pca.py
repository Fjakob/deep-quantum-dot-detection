import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

from src.lib.dimensionReduction.dim_reducer import DimReducer

class PCA(DimReducer):
    """ Principal Component Analysis class for dimensionality reduction. """
    def __init__(self, latent_dim=12):
        self.latent_dim = latent_dim
        self.mean = None
        self.singular_values = None
        self.V = None


    def fit(self, X_train):
        """ Fits PCA parameter to training data. """
        mean = np.mean(X_train, axis=0)
        X_train_centered = X_train - mean

        _, S, Vt = linalg.svds(X_train_centered, k=self.latent_dim)

        self.singular_values = np.flip(S)
        self.V = np.transpose(Vt)
        self.mean = mean


    def reduce(self, X, return_reconstruction=False):
        """ Reduces input matrix X to lower dimensional latent representation. """
        if self.V is None:
            print("First fit model to a dataset.")
            return

        # center data and reduce        
        X_c = X - self.mean
        Z = np.matmul(X_c, self.V)

        if return_reconstruction:
            X_recon = self.mean + np.matmul(Z, np.transpose(self.V))
            return Z, X_recon
        return Z


    def plot_principal_components(self):
        if self.singular_values is None:
            print("First fit model to a dataset.")
            return
        plt.stem(self.singular_values / np.max(self.singular_values))
        plt.ylabel("Significance")
        plt.xlabel("Component")
        plt.show()


    