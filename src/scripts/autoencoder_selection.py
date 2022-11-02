from config.imports import *

from scipy.stats import spearmanr, pearsonr

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.feature_extractor import FeatureExtracter
from src.lib.neuralNetworks.dense_networks import VanillaNeuralNetwork as NeuralNetwork

from feature_selection import load_dataset


def main():

    retrain=True

    if retrain:
        log = dict()

        ### setup
        dataset_path  = 'datasets\labeled\data_w30_labeled.pickle'
        data_settings = {'add_artificial': False,
                         'artificial_samples': 20,
                         'max_artificial_peak_height': 15}

        hyperparams   = {'hidden_neurons': 100,
                         'dropout':        0.0,
                         'batch_size':     64,
                         'learning_rate':  0.01,
                         'epochs':         150,
                         'self_normalize': False,
                         'epsilon':        0.02}

        neural_net = NeuralNetwork(hyperparams)
        feature_extracter = FeatureExtracter(None, None, seed=42)
        feature_extracter.set_features('reconstruction_error')

        X, Y = load_dataset(dataset_path, data_settings)


        ### test latent dimensions
        latent_dims = [8, 16, 24, 32, 48, 64]
        scores    = list()
        pearsons  = list()
        spearmans = list()

        for latent_dim in latent_dims:
            
            reconstructor = Autoencoder(latent_dim)
            reconstructor.load_model(f'models/autoencoders/epsilon_1e-12/autoencoder{latent_dim}.pth')
            feature_extracter.reconstructor = reconstructor

            score = feature_extracter.evaluate_features((X,Y), neural_net, folds=5)
            scores.append(score)

            V = np.squeeze(feature_extracter.extract_from_dataset(X))

            pearsons.append(pearsonr(V, Y).statistic)
            spearmans.append(spearmanr(V, Y).correlation)

        log = {'Latent_dim': latent_dims,
               'Scores':     scores,
               'Pearsons':   pearsons,
               'Spearmans':  spearmans}

        with open(os.path.join('reports',f'autoencoder_selection_log.pickle'), 'wb') as file:
            pickle.dump(log, file)
    
    else:

        with open(os.path.join('reports',f'autoencoder_selection_log.pickle'), 'rb') as file:
            log = pickle.load(file)


    ### create statistic graphics
    latent_dims = log['Latent_dim']
    scores      = log['Scores']
    pearsons    = log['Pearsons']
    spearmans   = log['Spearmans']

    ### Use Latex text interpreter
    plt.rcParams['text.usetex'] = True

    ### Cross Validation Scores
    plt.figure(figsize=(6,4), dpi=1200)
    plt.subplot(2,1,1)
    plt.plot(latent_dims, scores, '-*')
    plt.ylabel('Cross Validation Score')
    plt.grid()

    ### Spearman Correlation
    plt.subplot(2,1,2)
    plt.plot(latent_dims, np.abs(spearmans), '-*')
    plt.xlabel('Latent dimension')
    plt.ylabel('Spearman Correlation')
    plt.grid()

    plt.suptitle('Label - $e_r$ interaction')
    plt.savefig('autoencoder_selection.png')



if __name__ == "__main__":
    main()