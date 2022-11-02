from json import load
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
from src.lib.classifiers.feature_based import SpectrumClassifier


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


def load_reconstructor(reconstructor, reconstr_settings):
    """ Loads saved model into a new autoencoder instance. """
    model_dir  = reconstr_settings['model_path']
    latent_dim = reconstr_settings['latent_dim']
    epsilon    = reconstr_settings['epsilon']

    file_ending = 'pth'
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
    dataset_path      = config['data']['path_data_augmented']
    val_data_path     = config['data']['path_validation_data']
    data_settings     = config['data']['data_setting']
    reconstr_type     = config['reconstructor']['type']
    reconstr_setting  = config['reconstructor'][reconstr_type]
    regressor_setting = config['neural_net']
    plots_settings    = config['results']['plot_settings']
    model_save_path   = config['results']['model_path']

    ### Pipeline stages
    # 0) instanciation
    reconstructor = load_reconstructor(reconstr_type, reconstr_setting)
    #peak_detector = OS_CFAR(N=190, T=6.9, N_protect=20)
    feature_extracter = FeatureExtracter(None, reconstructor, seed)
    neural_net = NeuralNetwork(hyperparams=regressor_setting)

    # 1) data processing stage
    X, Y = load_dataset(dataset_path, data_settings)
    X_val, Y_val = load_dataset(val_data_path)

    feature_extracter.set_features(['latent_representation'])
    
    V = feature_extracter.extract_from_dataset(X, rescale=True)
    V_val = feature_extracter.extract_from_dataset(X_val, rescale=False)

    # 2) training stage
    neural_net.fit(V, Y)

    """     v = np.linspace(0,10)
    y = neural_net.predict(v)
    plt.plot(V, Y, 'r*')
    plt.plot(V_val, Y_val, 'g*')
    plt.plot(v, y, 'b')
    plt.show()
    """
    # # 3) validation stage
    spectrum_rater = SpectrumClassifier(feature_extracter, neural_net)
    score = spectrum_rater.score(X_val, Y_val)
    print(f"Score: {score}")

    results = {spectrum_rater.rate(x)[0][0]: x for x in X_val}
    sorted_spectra = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}

    for rating, spectrum in sorted_spectra.items():
         plt.plot(spectrum)
         plt.title("Ranked as {:.2f}".format(rating))
         plt.show()

    for idx in range(20):
         x_val, y_val = X_val[idx], Y_val[idx]
         v = feature_extracter.extract_from_dataset(x_val)
         y_pred = neural_net.predict(v)
         plt.plot(x_val)
         plt.title(f"True: {y_val}, Predicted: {y_pred[0][0]}")
         plt.show()
    #with open(f"{model_save_path}//feature_based_rater.pickle", 'wb') as f:
    #    pickle.dump(spectrum_rater, f)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    pipeline(config_path=args.config)
