from config.imports import *

from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA


def load_autoencoder(latent_dim=12):
    """ Loads saved model into a new autoencoder instance. """

    model_path = f"models/autoencoders/autoencoder{latent_dim}.pth"

    if not os.path.isfile(model_path):
        print(f"No autoencoder model found for latent dimension {latent_dim}.\n")
        print("Availabel models:\n" + str([os.path.basename(element) for element in glob.glob("models\\autoencoders\\*.pth")]))
        exit()

    autoencoder = Autoencoder(latent_dim)
    autoencoder.load_model(model_path, model_summary=False)

    return autoencoder


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


def validate(X_pca, X, latent_dims):
    
    test_loss = []
    for latent_dim in latent_dims:
        print(f"Test loss for latent dimension {latent_dim}: \n")
        autoencoder = load_autoencoder(latent_dim)
        pca = PCA(latent_dim)
        pca.fit(X_pca)

        X_norm, _, X_hat = autoencoder.normalize_and_extract(X)
        _, _, X_hat_pca = pca.normalize_and_extract(X)

        error_autoencoder = np.linalg.norm(X_norm - X_hat)
        error_pca = np.linalg.norm(X_norm - X_hat_pca)

        print(f"Autoencoder: {error_autoencoder}, PCA: {error_pca} \n")
        test_loss.append((error_autoencoder, error_pca))
        

def main():

    dim_ae  = 256
    dim_pca = 256

    autoencoder = load_autoencoder(latent_dim = dim_ae)
    pca = PCA(latent_dim = dim_pca)

    # fit PCA to whole dataset
    with open('dataSets/unlabeled/data_w30_unlabeled_normalized_augmented.pickle', 'rb') as f:
        X_train = np.asarray(pickle.load(f))
    pca.fit(X_train)

    #load datasets
    with open('dataSets/labeled/data_w30_labeled.pickle', 'rb') as f:
       X, _ = pickle.load(f)
    np.random.shuffle(X)

    X_norm, _, X_hat = autoencoder.normalize_and_extract(X)
    _, _, X_hat_pca = pca.normalize_and_extract(X)

    plot_comparison(X_norm, X_hat, X_hat_pca, plots=15)



if __name__ == '__main__':
    main()