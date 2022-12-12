from config.imports import *

from sklearn.utils import shuffle

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork

import warnings
warnings.filterwarnings("ignore")


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


def select_explainable_features():

    mode = 'forward_selection'
    #mode = 'backward_elimination'

    ### set false to display obtained results saved into log
    retrain = True

    if retrain:
        ### setup
        dataset_path  = 'datasets\labeled\data_w30_labeled.pickle'
        data_settings = {'add_artificial': False,
                         'artificial_samples': 15,
                         'max_artificial_peak_height': 15}

        hyperparams   = {'hidden_neurons': 100,
                         'dropout':        0.0,
                         'batch_size':     64,
                         'learning_rate':  0.01,
                         'epochs':         100,
                         'self_normalize': False,
                         'epsilon_pred':   1e-12,
                         'var_prediction': False}

        latent_dim = 16

        ### preparation
        reconstructor = Autoencoder(latent_dim, epsilon=1e-12)
        peak_detector = OS_CFAR(N=193, T=6.54, N_protect=25)
        neural_net = NeuralNetwork(hyperparams)

        X, Y = load_dataset(dataset_path, data_settings)
        reconstructor.load_model(f'models/autoencoders/autoencoder{latent_dim}.pth')
        feature_extracter = FeatureExtracter(peak_detector, reconstructor, seed=42)

        ### selection algorithm
        if mode == 'forward_selection':
            log = feature_extracter.feature_forward_selection((X, Y), neural_net)
        elif mode == 'backward_elimination':
            log = feature_extracter.feature_backward_elimination((X, Y), neural_net)
        else:
            print('Unvalid selection algorithm.')

        ### logging
        with open(os.path.join('reports',f'{mode}_log.pickle'), 'wb') as file:
            pickle.dump(log, file)
    
    else:
        with open(os.path.join('reports',f'{mode}_log.pickle'), 'rb') as file:
            log = pickle.load(file)


    ### Create Nice Plot
    title = 'Feature Forward Selection' if mode=='forward_selection' else 'Sequential Backward Elimination'

    color_map = {'reconstruction_error': 'red',
                 'n_peak':               'blue',
                 "d_min":                'pink',
                 "w_max":                'green',
                 "min_to_max":           'orange',  
                 "x_max":                'purple',
                 "noise_correlation":    'cyan'}

    max_perf = list()
    for iter, performance_dict in log.items():
        max_perf.append([iter, max(performance_dict.values())])
        performance_dict = dict(sorted(performance_dict.items(), key=lambda item: item[1]))
        for feature, performance in performance_dict.items():
            color = color_map[feature]
            plt.plot(iter, performance, '*', color=color, label=feature, markersize=8)
        if iter == 1:
            plt.legend()

    plt.plot(np.asarray(max_perf)[:,0], np.asarray(max_perf)[:,1], 'k', zorder=1)

    plt.grid()
    plt.ylim((0.45, 1))
    plt.xlabel('Iteration')
    plt.ylabel('R2 score')
    plt.title(title)
    plt.show()


def select_latent_features():
    ### setup
    dataset_path  = 'datasets\labeled\data_w30_labeled.pickle'
    data_settings = {'add_artificial': True,
                    'artificial_samples': 15,
                    'max_artificial_peak_height': 15}

    # (good hyperparams for backward elimination, adjust for forward selection)
    # (add 20 artificials)
    hyperparams   = {'hidden_neurons': 50,
                    'dropout':        0.0,
                    'batch_size':     64,
                    'learning_rate':  0.01,
                    'epochs':         100,
                    'self_normalize': False,
                    'epsilon_pred':   1e-12,
                    'var_prediction': False}

    neural_net = NeuralNetwork(hyperparams)

    X, Y = load_dataset(dataset_path, data_settings)

    log = list()
    latent_dims = [8,16,24,32,48,64]
    for latent_dim in latent_dims:

        print(f"Loading model p={latent_dim}.")
        reconstructor = Autoencoder(latent_dim, epsilon=1e-12)
        reconstructor.load_model(f'models/autoencoders/autoencoder{latent_dim}.pth')
        feature_extracter = FeatureExtracter(None, reconstructor, seed=42)
        feature_extracter.set_features('latent_representation')
        r2 = feature_extracter.evaluate_features(eval_dataset=(X, Y), regressor=neural_net, folds=5)

        log.append(r2)
        print(f'R2: {r2:.3f}\n')

    plt.figure()
    plt.plot(latent_dims, log,'-*')
    plt.xlabel('Latent dim')
    plt.ylabel('R2 score')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    select_explainable_features()
    #select_latent_features()

