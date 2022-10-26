from config.imports import *
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork


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

    X, Y = shuffle(X, Y)
    Y = np.ravel(Y)
    
    return X, Y


def load_reconstructor(reconstructor, reconstr_settings):
    """ Loads saved model into a new autoencoder instance. """
    model_dir  = reconstr_settings['model_path']
    latent_dim = reconstr_settings['latent_dim']

    file_ending = 'pth' if reconstructor == 'autoencoder' else 'pickle'
    model_path  = f"{model_dir}/{reconstructor}{latent_dim}.{file_ending}"
    model = Autoencoder(latent_dim) if reconstructor == 'autoencoder' else PCA(latent_dim)
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
    dataset_path      = config['data']['path_data_augmented']
    data_settings     = config['data']['data_setting']
    reconstr_type     = config['reconstructor']['type']
    reconstr_setting  = config['reconstructor'][reconstr_type]
    regressor_type    = config['regressor']['type']
    regressor_setting = config['regressor'][regressor_type]
    plots_settings    = config['results']['plot_settings']
    model_save_path   = config['results']['model_path']

    ### Pipeline stages
    # 0) instanciation
    reconstructor = load_reconstructor(reconstr_type, reconstr_setting)
    peak_detector = OS_CFAR(N=190, T=6.9, N_protect=20)
    feature_extracter = FeatureExtracter(peak_detector, reconstructor, seed)
    neural_net = NeuralNetwork(hyperparams=regressor_setting)

    # 1) data processing stage
    X, Y = load_dataset(dataset_path, data_settings)
    
    feature_extracter.feature_forward_selection((X,Y), neural_net)
    #feature_extracter.set_features(['reconstruction_error', 'n_peak', 'min_to_max'])
    #V = feature_extracter.extract_from_dataset(X, rescale=True)

    # 2) training stage
    #V_train, V_test, Y_train, Y_test = train_test_split(V, Y, test_size=0.2, random_state=seed)
    #neural_net.fit(V_train, Y_train, print_training_curve=True, self_normalizing=False)

    #r2_test = neural_net.score(V_test, Y_test)
    #print(f"Test R2 score: {r2_test}")

    # 3) evaluation
    #save_model(model_save_path, feature_extracter, scaler, gp_regressor)
    #visualize_deploy(X, Y, feature_extracter, scaler, gp_regressor)



if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    pipeline(config_path=args.config)
