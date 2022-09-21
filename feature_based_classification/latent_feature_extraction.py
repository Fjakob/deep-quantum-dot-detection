import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torchsummary import summary
from lib.dimensionReduction.autoencoder import autoencoder
from lib.dimensionReduction.pca import PCA


def load_autoencoder(latent_dim=12):
    """ Loads saved model into a new autoencoder instance. """

    model_path = 'autoencoders/autoencoder{}.pth'.format(latent_dim)

    model_autoencoder = autoencoder(latent_dim).to('cuda')
    model_autoencoder.load_state_dict(torch.load(model_path))

    summary(model_autoencoder, (1,1024))
    return model_autoencoder


def load_dataset(PATH):
    # load the dataset
    with open(PATH, 'rb') as f:
        dataset = pickle.load(f)
    X = np.asarray([x for _, x, _ in dataset])
    Y = np.asarray([y for _, _, y in dataset])
    return X, Y


def compare_dimension_reduction(dim_ae=12, dim_pca=128):
    # load dimension reducer
    model_autoencoder = load_autoencoder(latent_dim = dim_ae)
    model_pca = PCA(latent_dim = dim_pca)

    # fit PCA to whole dataset
    with open('dataSets/DataFilteredNormalizedAugmented', 'rb') as f:
        X_train = np.asarray(pickle.load(f))
    model_pca.fit(X_train)

    # load datasets
    X, _ = load_dataset('dataSets/regressionData')
    np.random.shuffle(X)

    X_norm, Z, X_hat = model_autoencoder.reduce_raw(X)
    Z_pca, X_hat_pca = model_pca.reduce(X_norm, return_reconstruction=True)

    plot_comparison(X_norm, X_hat, X_hat_pca, plots=15)


def plot_comparison(X, X_recon, X_recon_pca, plots=20):
    for idx in range(plots):
        x = X[idx,:]
        x_recon = X_recon[idx,:]
        x_recon_pca = X_recon_pca[idx,:]
        ymin = np.min(x) if np.min(x) < np.min(x_recon) else np.min(x_recon)
        ymax = np.max(x) if np.max(x) > np.max(x_recon) else np.max(x_recon)
        plt.figure()
        ax = plt.subplot(2,1,1)
        ax.set_ylim((ymin, ymax))
        ax.set_title("PCA")
        ax.plot(x)
        ax.plot(x_recon_pca,'--')
        ax = plt.subplot(2,1,2)
        ax.set_ylim((ymin, ymax))
        ax.set_title("Autoencoder")
        ax.plot(x)
        ax.plot(x_recon,'--')
    plt.show()


if __name__ == '__main__':
    compare_dimension_reduction(dim_ae=128, dim_pca=128)
    

