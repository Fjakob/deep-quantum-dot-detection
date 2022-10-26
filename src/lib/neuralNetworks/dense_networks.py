import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, tensor, cuda, optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score


class VanillaNeuralNetwork():
    """ One Layer Network with Sigmoid activation function. """
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.network = None

    
    def network_architecture(self, input_dim, hidden_neurons, p_drop, self_normalize):
        
        if self_normalize:
            hidden_layer = nn.Linear(input_dim, hidden_neurons, bias=False)
            output_layer = nn.Linear(hidden_neurons, 1)
            activation   = nn.SELU()
            dropout      = nn.AlphaDropout(p_drop)
            nn.init.kaiming_normal_(hidden_layer.weight)
            nn.init.kaiming_normal_(output_layer.weight)
        else:
            hidden_layer = nn.Linear(input_dim, hidden_neurons, bias=True)
            output_layer = nn.Linear(hidden_neurons, 1)
            activation   = nn.Sigmoid()
            dropout      = nn.Dropout(p_drop)

        network = nn.Sequential(hidden_layer,
                                activation,
                                dropout,
                                output_layer,
                                nn.Sigmoid()) 
        return network


    def fit(self, X_train, Y_train, self_normalizing=False, verbose=True, print_training_curve=False):
        """ Trains a one-layer neural network with modular input data dimension. """

        ### Load hyperparameters
        hidden_neurons = self.hyperparams['hidden_neurons']
        p_drop         = self.hyperparams['dropout']
        batch_size     = self.hyperparams['batch_size']
        learning_rate  = self.hyperparams['learning_rate']
        epochs         = self.hyperparams['epochs']

        ### Extract dimensions
        if X_train.shape == 1:
            X_train = np.expand_dims(X_train, axis=1)
        n_dim = X_train.shape[1]

        ### Data preparation
        X = tensor(X_train, device=self.device)
        Y = tensor(Y_train, device=self.device)
        dataset = TensorDataset(X.float(), Y.float())
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        ### Model initialization
        network = self.network_architecture(n_dim, hidden_neurons, p_drop, self_normalizing)
        network.to(self.device)

        ### Model training
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss() #nn.L1Loss() #nn.HuberLoss()
        losses = []
        if verbose:
            print("Training neural network...")
        for epoch in range(epochs):
            loss = 0
            for x, y in train_loader:
                x = torch.unsqueeze(x, dim=1)
                optimizer.zero_grad() 
                y_pred = network(x)
                y_pred = torch.reshape(y_pred, y.size())
                train_loss = loss_function(y, y_pred)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(train_loader)
            losses.append(loss)

        self.network = network

        if verbose:
            print(f"Training R2 score: {self.score(X_train ,Y_train)}")

        if print_training_curve:
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Train Loss')
            plt.show()


    
    def predict(self, X):
        """ Computes Neural Network Output for given Input. """

        if X.shape == 1:
            X = np.expand_dims(X, axis=1)

        X = tensor(X, device=self.device).float()
        Y_pred = self.network(X)
        return Y_pred.cpu().detach().numpy()

    
    def score(self, X, Y):
        """ Computes R2-score for given ground truth. """

        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)



