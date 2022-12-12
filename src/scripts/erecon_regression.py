from config.imports import *
import argparse
import yaml
import warnings
import random
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork

import tikzplotlib


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
        print('Adding artificial data')
        loader = DataProcesser(absolute_database_path=None, spectrum_size=1024)
        X_art = loader.create_artificial_spectra(n_samples=artificial_samples, max_peak_height=max_peak_height)
        Y_art = np.zeros(X_art.shape[0])

        X = np.vstack((X, X_art)) 
        Y = np.hstack((Y, Y_art))

    X, Y = shuffle(X, Y, random_state=42)
    Y = np.ravel(Y)
    
    return X, Y


def load_autoencoder(reconstr_settings):
    """ Loads saved model into a new autoencoder instance. """
    model_dir  = reconstr_settings['model_path']
    latent_dim = reconstr_settings['latent_dim']
    epsilon    = reconstr_settings['epsilon']

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
    
    seed                = config['base']['seed']
    dataset_path        = config['data']['path_data']
    data_settings       = config['data']['data_setting']
    autoencoder_setting = config['reconstructor']['autoencoder']
    network_params      = config['neural_net']['probabilistic']

    ### Pipeline stages
    # 0) instanciation
    random.seed(42)
    np.random.seed(42)
    autoencoder = load_autoencoder(autoencoder_setting)
    feature_extractor = FeatureExtracter(None, reconstructor=autoencoder, seed=seed)
    feature_extractor.set_features(['reconstruction_error'])
    neural_net = NeuralNetwork(hyperparams=network_params)

    # 1) data processing / feature extraction
    X, Y = load_dataset(dataset_path, data_settings)
    V = feature_extractor.extract_from_dataset(X, rescale=False, )

    # 2) training stage
    neural_net.fit(V, Y, print_training_curve=True)

    # 3) validation stage
    v = np.linspace(0,10)
    y, sig = neural_net.predict(v, return_var=True)
    plt.figure()
    plt.plot(V, Y, 'r*', label='Data')
    plt.plot(v, y, label='Prediction')
    plt.fill_between(
         x=v,
         y1 = y + 2*sig,
         y2 = y - 2*sig,
         alpha=0.2,
         color="black",
         label="95% conf")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    pipeline(config_path=args.config)