import numpy as np
import pickle
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

from src.lib.featureExtraction.latent_extractor import LatentExtracter


class PCA(LatentExtracter):
    """ Principal Component Analysis class for dimensionality reduction. """
    def __init__(self, latent_dim=12):
        self.latent_dim = latent_dim
        self.mean = None
        self.V = None


    def fit(self, X_train):
        """ Fits PCA parameter to training data. """
        mean = np.mean(X_train, axis=0)
        X_train_centered = X_train - mean

        _, _, Vt = linalg.svds(X_train_centered, k=self.latent_dim)

        self.V = np.transpose(Vt)
        self.mean = mean


    def load_model(self, model_path, model_summary=False):
        """ Loads PCA matrices into instance. """
        with open(model_path, 'rb') as f:
            self.V, self.mean = pickle.load(f)
        if self.V.shape[1] != self.latent_dim:
            print(self.V.shape)
            print(self.latent_dim)
            raise ValueError("Loaded model doesn't fit object latent dimension.")
        if model_summary:
            print(f'Loaded PCA instance with latent dimension {self.V.shape[1]}')


    def extract_latent(self, X, return_reconstruction=False):
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

    