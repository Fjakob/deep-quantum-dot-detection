from __config__ import *

import torch
from os.path import isfile

from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA


def load_autoencoder(latent_dim=12):
    """ Loads saved model into a new autoencoder instance. """

    model_path = f"models/autoencoders/autoencoder{latent_dim}.pth"

    if not isfile(model_path):
        print(f"No autoencoder model found for latent dimension {latent_dim}.\n")
        print("Availabel models:\n" + str(os.listdir("models/autoencoders")))
        exit()

    autoencoder = Autoencoder(latent_dim)
    autoencoder.load_model(model_path)

    return autoencoder


def load_dataset(PATH):
    # load the dataset
    with open(PATH, 'rb') as f:
        dataset = pickle.load(f)
    X = np.asarray([x for _, x, _ in dataset])
    Y = np.asarray([y for _, _, y in dataset])
    return X, Y


def plot_comparison(X, X_recon, X_recon_pca, plots=20):
    for idx in range(plots):
        x = X[idx,:]
        x_recon = X_recon[idx,:]
        x_recon_pca = X_recon_pca[idx,:]

        # set axis limits
        ymin = np.min(x) if np.min(x) < np.min(x_recon) else np.min(x_recon)
        ymax = np.max(x) if np.max(x) > np.max(x_recon) else np.max(x_recon)

        # create plot
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


def main():

    dim_ae  = 24
    dim_pca = 24

    autoencoder = load_autoencoder(latent_dim = dim_ae)
    pca = PCA(latent_dim = dim_pca)

    # fit PCA to whole dataset
    with open('dataSets/DataFilteredNormalizedAugmented', 'rb') as f:
        X_train = np.asarray(pickle.load(f))
    pca.fit(X_train)

    # load datasets
    X, _ = load_dataset('dataSets/regressionData')
    np.random.shuffle(X)

    X_norm, _, X_hat = autoencoder.normalize_and_reduce(X)
    _, _, X_hat_pca = pca.normalize_and_reduce(X)

    plot_comparison(X_norm, X_hat, X_hat_pca, plots=15)

if __name__ == '__main__':
    main()