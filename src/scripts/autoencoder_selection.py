from config.imports import *

from scipy.stats import spearmanr, pearsonr

from src.lib.utils.norms import window_loss
from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA
from src.lib.featureExtraction.feature_extractor import FeatureExtracter

from sklearn.utils import shuffle

def load_dataset(path, artif_data_setup):
    """ Loads dataset from saved file. """
    add_artificial = artif_data_setup['add_artificial']
    artificial_samples = artif_data_setup['artificial_samples']
    max_peak_height = artif_data_setup['max_artificial_peak_height']

    with open(path, 'rb') as file:
        X, Y = pickle.load(file)
    
    if add_artificial:
        loader = DataProcesser(absolute_database_path=None, spectrum_size=1024)
        X_art = loader.create_artificial_spectra(n_samples=artificial_samples, max_peak_height=max_peak_height)
        Y_art = np.zeros(X_art.shape[0])

        X = np.vstack((X, X_art)) 
        Y = np.hstack((Y, Y_art))

    X, Y = shuffle(X, Y, random_state=42)
    Y = np.ravel(Y)
    
    return X, Y


def spearman():

    with open('dataSets/unlabeled/data_w30_unlabeled_normalized_augmented.pickle', 'rb') as f:
        X_train = np.asarray(pickle.load(f))

    ### setup
    dataset_path  = 'datasets\labeled\data_w30_labeled.pickle'
    data_settings = {'add_artificial': False,
                        'artificial_samples': 15,
                        'max_artificial_peak_height': 15}

    X, Y = load_dataset(dataset_path, data_settings)

    ### test latent dimensions
    latent_dims = [8, 16, 24, 32, 48, 64]
    spearmans = list()

    for latent_dim in latent_dims:
        print(f'Latent dim {latent_dim}')
        
        autoencoder = Autoencoder(latent_dim)
        autoencoder.load_model(f'models/autoencoders/autoencoder{latent_dim}.pth')

        pca = PCA(latent_dim)
        pca.fit(X_train)

        idx=0
        O = list()
        for x in X:
            y = Y[idx]
            idx+=1

            x_1, _, x_r1 = autoencoder.normalize_and_extract(x)
            e_recon = window_loss(x_1, x_r1, window_size=5)
            x_1, _, x_r1 = pca.normalize_and_extract(x)
            e_recon_pca = window_loss(x_1, x_r1, window_size=5)

            if e_recon < 2.3:
                O.append([e_recon, e_recon_pca, y])

        O = np.asarray(O)
        corr = spearmanr(O[:,0], O[:,2]).correlation
        corr_pca = spearmanr(O[:,1], O[:,2]).correlation
        print(f'Correlation = {corr:.2f} (ae), {corr_pca:.2f} (pca)')
        spearmans.append([corr, corr_pca])

        # if latent_dim == 16:
        #     plt.figure()
        #     plt.title(f'Latent dim {latent_dim}')
        #     plt.plot(O[:,0], O[:,2], 'r*')
        #     plt.xlabel('e_recon')
        #     plt.ylabel('y')
        #     plt.show()

    spearmans = np.asarray(spearmans)
    plt.figure()
    plt.plot(latent_dims, np.abs(spearmans[:,0]), '*--b', label='Autoencoder')
    plt.plot(latent_dims, np.abs(spearmans[:,1]), '*--r', label='PCA')
    plt.grid()
    plt.xlabel('Latent dim')
    plt.ylabel('Spearman')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    spearman()