from config.imports import *
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.featureExtraction.peak_feature_extracter import PeakFeatureExtracter
from src.lib.classifiers.intuitive_feature_based import PeakFeatureBasedRater


def load_dataset(path, artif_data_setup):
    """ Loads dataset from saved file. """
    add_artificial = artif_data_setup['add_artificial']
    artificial_samples = artif_data_setup['artificial_samples']
    max_peak_height = artif_data_setup['max_peak_height']

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


def train_gaussian_process(dataset, gp_kernel):
    V_train, V_test, Y_train, Y_test = dataset

    kernel_list = {"Linear": kernels.DotProduct() + kernels.WhiteKernel(), 
                   "Gaussian": kernels.RBF(),
                   "Matern": kernels.Matern(nu=1/2),
                   "Rational quadratic": kernels.RationalQuadratic()}
    kernel = kernel_list[gp_kernel]
    
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(V_train, Y_train)

    print("Training R2-score: {}".format(gpr.score(V_train,Y_train)))
    print("Test R2-score: {}".format(gpr.score(V_test,Y_test)))

    return gpr


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
    
    seed             = config['base']['seed']
    dataset_path     = config['data']['path_data_augmented']
    artif_data_setup = config['data']['artificial_data_setting']
    gp_kernel        = config['pipeline_i']['gaussian_process']['kernel']
    plot_path        = config['results']['plot_path']
    plot_format      = config['results']['figure_format']
    model_save_path  = config['results']['model_path']

    ### Pipeline stages
    # 1) data processing stage
    X, Y = load_dataset(dataset_path, artif_data_setup)
    feature_extracter = PeakFeatureExtracter(peak_detector=OS_CFAR(N=190, T=6.9, N_protect=20), seed=seed)
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

