import numpy as np
import matplotlib.pyplot as plt


class FeatureExtracter():
    """ Abstract class for dimension reduction. """

    def extract_latent(self, X_normalized, return_reconstruction=False):
        raise NotImplementedError


    def normalize_and_extract(self, X):
        """ Scales raw numpy spectrum to value range [0,1] before reducing. """
        X_scaled = X / np.max(np.abs(X), axis=1)[:,np.newaxis]
        Z, X_recon = self.extract_latent(X_scaled, return_reconstruction=True)
        return X_scaled, Z, X_recon


    def plot_reconstructions(self, X, nbr_plots=5):
        """ Visualizes dimension reducers capability by comparing reconstruction with corresponding input. """
        np.random.shuffle(X)
        _, X_recon = self.extract_latent(X, return_reconstruction=True)
        for idx in range(nbr_plots):
            x = X[idx,:]
            x_recon = X_recon[idx,:]
            ymin = np.min(x) if np.min(x) < np.min(x_recon) else np.min(x_recon)
            ymax = np.max(x) if np.max(x) > np.max(x_recon) else np.max(x_recon)
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(x)
            ax.set_ylim((ymin, ymax))
            ax.plot(x_recon, '--')
        plt.show()
