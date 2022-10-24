from config.imports import *
import argparse
import yaml
import torch
from torch import nn, tensor, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.featureExtraction.pca import PCA
from src.lib.classifiers.reconstruction_based import ReconstructionBasedRater
from src.lib.dataProcessing.data_processer import DataProcesser
from src.lib.peakDetectors.os_cfar import OS_CFAR


def signal_window(x, idx, shift):
    idx_left = idx - shift
    idx_right = idx + shift + 1

    # take signal window
    if idx_left < 0:
        padding_left = np.zeros( abs(idx_left) )
        x_window = np.concatenate( (padding_left, x[0:idx_right]) )
    elif idx_right > len(x):
        padding_right = np.zeros( idx_right - len(x) )
        x_window = np.concatenate( (x[idx_left:len(x)], padding_right) )
    else:
        x_window = x[idx_left:idx_right]

    return x_window


def window_loss(X1, X2, window_size=9):
    if len(X1.shape) == 1:
        X1 = np.expand_dims(X1, axis=0)
    if len(X2.shape) == 1:
        X2 = np.expand_dims(X2, axis=0)
    n = X1.shape[0]
    shift = int((window_size-1)/2)

    loss = []
    idx=0
    for idx in range(n):
        x1 = X1[idx]
        x2 = X2[idx]
        diff = x1-x2
        e = []
        for idx in range(len(diff)):
            window = signal_window(diff, idx, shift)
            e.append(np.mean(window))
        loss.append(np.linalg.norm(e))
    return np.asarray(loss)


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


def compute_reconstruction_errors(spectra, dim_reducer):
    """ Compute reconstruction error of spectra """
    X_norm, _, X_hat = dim_reducer.normalize_and_extract(spectra)
    #reconstruction_errors = np.linalg.norm(X_norm - X_hat, axis=1)
    #reconstruction_errors = shape_loss(X_hat, X_norm)
    reconstruction_errors = window_loss(X_hat, X_norm)
    return reconstruction_errors

def detect_peaks(spectra, peak_detector):
    N = []
    for spectrum in spectra:
        _, n_peaks, _ = peak_detector.detect(spectrum)
        N.append(n_peaks)
    return np.asarray(N)


def visualize_data(X_e, Y, artif_data_setup):
    add_artificial = artif_data_setup['add_artificial']
    artificial_samples = artif_data_setup['artificial_samples']
    n_data = X_e.shape[0] - artificial_samples
 
    if add_artificial:
        plt.plot(X_e[:n_data], Y[:n_data], 'b*', label='Nom. data')
        plt.plot(X_e[n_data:], Y[n_data:], 'r*', label='Art. data')
    else:
        plt.plot(X_e, Y, 'b*', label='Data')
    plt.legend()
    plt.show()

def visualize_data_3d(X_e, N, Y, artif_data_setup):
    add_artificial = artif_data_setup['add_artificial']
    artificial_samples = artif_data_setup['artificial_samples']
    n_data = X_e.shape[0] - artificial_samples
 
    if add_artificial:
        plt.scatter(X_e[:n_data], N[:n_data], Y[:n_data], label='Nom. data')
        plt.scatter(X_e[n_data:], N[n_data:], Y[n_data:], label='Artificial data')
    else:
        plt.scatter(X_e[:], N[:], Y[:], label='Data')
    plt.legend()
    plt.show()


def train_neural_network(errors, peaks, labels, hyperparams):
    """ Trains a one-layer neural network. """

    ### Load hyperparameters
    hidden_neurons = hyperparams['hidden_neurons']
    p_drop         = hyperparams['dropout']
    batch_size     = hyperparams['batch_size']
    learning_rate  = hyperparams['learning_rate']
    epochs         = hyperparams['epochs']

    ### Setup device
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    ### Data preparation
    X = np.transpose(np.vstack((errors,peaks)))
    X = tensor(X, device=device)
    Y = tensor(labels, device=device)
    dataset = TensorDataset(X.float(), Y.float())
    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    ### Model initialization
    network = nn.Sequential(nn.Linear(2, hidden_neurons), 
                            nn.Sigmoid(), 
                            nn.Dropout(p_drop), 
                            nn.Linear(hidden_neurons, 1),
                            nn.Sigmoid())
    network.to(device)

    ### Model training
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.L1Loss() #nn.MSELoss() # #nn.HuberLoss()
    losses = []
    for epoch in tqdm(range(epochs)):
        loss = 0
        for e, y in train_loader:
            e = torch.unsqueeze(e, dim=1)
            optimizer.zero_grad() 
            y_pred = network(e)
            y_pred = torch.squeeze(y_pred)
            train_loss = loss_function(y, y_pred)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        losses.append(loss)

    R2 = r2_score(Y.cpu().detach().numpy(), network(X.float()).cpu().detach().numpy())
    print(f'R2-score: {R2}')
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.show()
    
    return network


def visualize(X, Y, network, plot_path, plot_format):
    """ Visualize network functionality """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.unsqueeze(torch.linspace(np.min(X), np.max(X), 100, device=device), dim=1)
    y = network(x)
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    saving_path = f'{plot_path}\pipeline_iv_results.{plot_format}'
    plt.plot(X, Y, 'b*', label='Data')
    plt.plot(x, y, 'r-', label='Prediction')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Rating')
    plt.legend()
    plt.savefig(saving_path)
    print(f'Saved result plot in {plot_path}.')


def save_model(model_dir, autoencoder, network):
    """ Save final model. """
    path = f'{model_dir}\\reconstruction_based_rater.pickle'
    rater = ReconstructionBasedRater(autoencoder, network)
    with open(path, 'wb') as file:
        pickle.dump(rater, file)
    print(f'Saved final model in {model_dir}.')


def visualize_deploy(X, Y, autoencoder, network):
    plt.close()
    rater = ReconstructionBasedRater(autoencoder, network)
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

    dataset_path     = config['data']['path_data']
    data_settings     = config['data']['data_setting']
    reconstr_type     = config['reconstructor']['type']
    reconstr_setting  = config['reconstructor'][reconstr_type]
    hyperparams      = config['regressor']['neural_net']
    """     plot_path        = config['results']['plot_path']
    plot_format      = config['results']['figure_format']
    model_save_path  = config['results']['model_path'] """


    ### Pipeline stages
    # 1) data processing stage
    X, Y = load_dataset(dataset_path, data_settings)
    reconstructer = load_reconstructor(reconstr_type, reconstr_setting)
    X_e = compute_reconstruction_errors(spectra=X, dim_reducer=reconstructer)
    #detector = OS_CFAR(N=190, T=6.9, N_protect=20)
    #N   = detect_peaks(spectra=X, peak_detector=detector) 
    visualize_data(X_e, Y, data_settings)

    # 2) training stage
    raise
    network = train_neural_network(errors=X_e, peaks=N, labels=Y, hyperparams=hyperparams)

    # 3) evaluation stage
    #visualize(X_e, Y, network, plot_path, plot_format)
    #save_model(model_save_path, reconstructer, network)
    #visualize_deploy(X, Y, reconstructer, network)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    pipeline(config_path=args.config)
