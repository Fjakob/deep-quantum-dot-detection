from config.__config__ import *
import argparse
import yaml
import torch
from torch import nn, tensor, optim
from torch.utils.data import TensorDataset, random_split, DataLoader

from src.lib.featureExtraction.autoencoder import Autoencoder
from src.lib.classifiers.reconstruction_based import ReconstructionBasedRater


def load_dataset(path):
    """ Loads dataset from saved file. """
    with open(path, 'rb') as file:
        X, Y = pickle.load(file)
    return X, Y


def load_autoencoder(model_dir, latent_dim=12):
    """ Loads saved model into a new autoencoder instance. """
    autoencoder = Autoencoder(latent_dim)
    model_path  = f"{model_dir}/autoencoder{latent_dim}.pth"
    autoencoder.load_model(model_path, model_summary=False)
    return autoencoder


def compute_reconstruction_errors(spectra, dim_reducer):
    """ Compute reconstruction error of spectra """
    X_norm, _, X_hat = dim_reducer.normalize_and_extract(spectra)
    reconstruction_errors = np.linalg.norm(X_norm - X_hat, axis=1)
    return reconstruction_errors


def train_neural_network(errors, labels, hyperparams):
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
    X = tensor(errors, device=device)
    Y = tensor(labels, device=device)
    dataset = TensorDataset(X.float(), Y.float())
    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    ### Model initialization
    network = nn.Sequential(nn.Linear(1, hidden_neurons), 
                            nn.Sigmoid(), 
                            nn.Dropout(p_drop), 
                            nn.Linear(hidden_neurons, 1))
    network.to(device)

    ### Model training
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss() #nn.HuberLoss()
    ###############################################
    # TODO: stage training loss
    ###############################################
    for epoch in range(epochs):
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
    
    return network


def visualize(X, Y, network, plot_path):
    """ Visualize network functionality """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.unsqueeze(torch.linspace(np.min(X), np.max(X), 100, device=device), dim=1)
    y = network(x)

    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    saving_path = f'{plot_path}\pipeline_iv_results.pdf'
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



def pipeline(config_path):
    
    ### Loading params
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    dataset_path     = config['data']['path_data']
    autoencoder_path = config['pipeline_iv']['autoencoder']['model_path']
    latent_dim       = config['pipeline_iv']['autoencoder']['latent_dim']
    hyperparams      = config['pipeline_iv']['neural_net']['hyperparams']
    plot_path        = config['results']['plot_path']
    model_path       = config['results']['model_path']

    ### Pipeline stages
    # 1) data processing stage
    X, Y = load_dataset(dataset_path)
    autoencoder = load_autoencoder(autoencoder_path, latent_dim)
    X_e = compute_reconstruction_errors(spectra=X, dim_reducer=autoencoder)

    # 2) training stage
    network = train_neural_network(errors=X_e, labels=Y, hyperparams=hyperparams)

    # 3) evaluation stage
    visualize(X_e, Y, network, plot_path)
    save_model(model_path, autoencoder, network)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    pipeline(config_path=args.config)

