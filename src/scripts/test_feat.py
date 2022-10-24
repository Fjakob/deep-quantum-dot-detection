from config.imports import *
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.peakDetectors.os_cfar import OS_CFAR


def load_reconstructor(reconstructor, reconstr_settings):
    """ Loads saved model into a new autoencoder instance. """
    model_dir  = reconstr_settings['model_path']
    latent_dim = reconstr_settings['latent_dim']

    file_ending = 'pth'
    model_path  = f"{model_dir}/{reconstructor}{latent_dim}.{file_ending}"
    model = Autoencoder(latent_dim)
    model.load_model(model_path)
    return model


def main():
    autoencoder = load_reconstructor('autoencoder', {'model_path': 'models/autoencoders', 'latent_dim': 32})
    peak_detector = OS_CFAR(N=20, T=6.9, N_protect=2)
    features = FeatureExtracter(peak_detector, autoencoder, seed=42)

    X = np.random.randn(200, 1024)

    V = features.extract_from_dataset(X)
    print(V.shape)

    print('Done')
   

if __name__ == "__main__":
    main()