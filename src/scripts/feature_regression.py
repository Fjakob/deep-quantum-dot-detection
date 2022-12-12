from config.imports import *
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork


def load_dataset(path, artif_data_setup=None):
    """ Loads dataset from saved file. """
    if artif_data_setup:
        add_artificial = artif_data_setup['add_artificial']
        artificial_samples = artif_data_setup['artificial_samples']
        max_peak_height = artif_data_setup['max_artificial_peak_height']
    else:
        add_artificial=False

    with open(path, 'rb') as file:
        X, Y = pickle.load(file)
    
    if add_artificial:
        loader = DataProcesser(absolute_database_path=None, spectrum_size=1024)
        X_art = loader.create_artificial_spectra(n_samples=artificial_samples, max_peak_height=max_peak_height)
        Y_art = np.zeros(X_art.shape[0])

        X = np.vstack((X, X_art)) 
        Y = np.hstack((Y, Y_art))

    X, Y = shuffle(X, Y)
    Y = np.ravel(Y)
    
    return X, Y


def load_autoencoder(autoencoder, autoenc_settings):
    """ Loads saved model into a new autoencoder instance. """
    model_dir  = autoenc_settings['model_path']
    latent_dim = autoenc_settings['latent_dim']
    epsilon    = autoenc_settings['epsilon']

    model_path  = f"{model_dir}/autoencoder{latent_dim}.pth"
    model = Autoencoder(latent_dim, epsilon)
    model.load_model(model_path)
    return model


##########################################################################################
#                                   P I P E L I N E
##########################################################################################

def pipeline(config_path):
    
    ### Loading params
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    
    seed              = config['base']['seed']
    dataset_path      = config['data']['path_data']
    data_settings     = config['data']['data_setting']
    reconstr_type     = config['reconstructor']['type']
    reconstr_setting  = config['reconstructor'][reconstr_type]
    regressor_setting = config['neural_net']['probabilistic']

    ### Pipeline stages
    # 0) instanciation
    autoencoder = load_autoencoder(reconstr_type, reconstr_setting)
    neural_net = NeuralNetwork(hyperparams=regressor_setting)
    feature_extracter = FeatureExtracter(OS_CFAR(N=193, T=6.54, N_protect=25), autoencoder, seed)
    feature_extracter.set_features(['reconstruction_error', 'min_to_max', 'x_max', 'n_peak'])
    #feature_extracter.set_features(["latent_representation"])
    
    # 1) data processing stage
    X, Y = load_dataset(dataset_path, data_settings)
    V = feature_extracter.extract_from_dataset(X, rescale=True)

    # 2) training stage
    neural_net.fit(V, Y, print_training_curve=True)

    # 3) validation stage
    for idx in range(20):
        x_val, y_val = X[idx], Y[idx]
        v = feature_extracter.extract_from_dataset(x_val)
        y_pred, sig = neural_net.predict(v, return_var=True)
        y_pred = max(0, min(1, y_pred[0]))
        conf_low = np.max([y_pred-2*np.abs(sig[0]), 0])
        conf_high = np.min([y_pred+2*np.abs(sig[0]), 1])
        plt.plot(x_val)
        plt.title("True: {:.2f}\n Predicted: {:.2f}\n 95% confidence: ({:.2f}, {:.2f})".format(y_val, y_pred, conf_low, conf_high))
        plt.show()

    
if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    pipeline(config_path=args.config)
