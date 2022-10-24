from config.imports import *
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.neural_network import MLPRegressor as neural_net

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA
from lib.featureExtraction.feature_extractor import FeatureExtracter
from lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork
from src.lib.classifiers.intuitive_feature_based import PeakFeatureBasedRater


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


def feature_transformation(X, feature_extractor, scaling=True):

    V = feature_extractor.extract_from_dataset(X)
    scaler = StandardScaler()

    if scaling:
        scaler.fit(V)
        V = scaler.transform(V)
        scale, mean = scaler.scale_, scaler.mean_
    else:
        scale, mean = 1, 0
    
    scaler = (scale, mean)
    return V, scaler


def save_model(model_dir, feature_extracter, scaler, regressor):
    path = f'{model_dir}\\intuitive_feature_based_rater.pickle'
    rater = PeakFeatureBasedRater(feature_extracter, scaler, regressor)
    with open(path, 'wb') as file:
        pickle.dump(rater, file)
    print(f'Saved final model in {model_dir}.')


def visualize_deploy(X, Y, feature_extracter, scaler, regressor):
    plt.close()
    rater = PeakFeatureBasedRater(feature_extracter, scaler, regressor)
    Y_pred = rater.rate(X)
    for _ in range(10):
        idx = np.random.randint(0, len(Y))
        x = X[idx]
        y = Y[idx]
        y_pred = Y_pred[idx]
        plt.plot(x)
        plt.title('Label: {:.2f}, Predicted: {:.2f}'.format(y,y_pred))
        plt.show()


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
    # 1) data processing stage
    X, Y = load_dataset(dataset_path, data_settings)

    peak_detector = OS_CFAR(N=190, T=6.9, N_protect=20)
    reconstructor = load_reconstructor(reconstr_type, reconstr_setting)
    feature_extracter = FeatureExtracter(peak_detector, reconstructor, seed)

    V = feature_extracter.extract_from_dataset(X)
    plt.plot(V, Y, '*')
    plt.show()
    
    neural_net = NeuralNetwork(regressor_setting)
    neural_net.fit(V, Y)


    raise

    #gpr = GaussianProcessRegressor(kernel=kernels.Matern(nu=1/2))
    #regressor = neural_net(hidden_layer_sizes=100, activation='logistic', alpha=0.1)
    feature_extracter.feature_backward_elimination(regressor, (X[0:150],Y[0:150]))

    raise
    V, scaler = feature_transformation(X, feature_extracter, scaling=True)
    dataset = train_test_split(V, Y, test_size=0.05)

    #gpr = GaussianProcessRegressor(kernel=kernels.Matern(nu=1/2))
    #feature_extracter.feature_selection(gpr, (X[0:150],Y[0:150]))

    # 2) training stage
    gp_regressor = train_gaussian_process(dataset, gp_kernel)

    # 3) evaluation
    save_model(model_save_path, feature_extracter, scaler, gp_regressor)
    visualize_deploy(X, Y, feature_extracter, scaler, gp_regressor)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    pipeline(config_path=args.config)

