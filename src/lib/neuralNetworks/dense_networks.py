import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import cuda, nn, optim, tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class VanillaNeuralNetwork():
    """ One Layer Network with Sigmoid activation function. """
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.network = None

        torch.manual_seed(42)

    
    def network_architecture(self, input_dim, hidden_neurons, variance_prediction=False, p_drop=0, self_normalize=False):

        torch.manual_seed(42)

        if variance_prediction:
            output_dim = 2
        else:
            output_dim = 1
        
        if self_normalize:
            hidden_layer = nn.Linear(input_dim, hidden_neurons, bias=False)
            output_layer = nn.Linear(hidden_neurons, output_dim)
            activation   = nn.SELU()
            dropout      = nn.AlphaDropout(p_drop)
            nn.init.kaiming_normal_(hidden_layer.weight)
            nn.init.kaiming_normal_(output_layer.weight)
        else:
            input_layer   = nn.Linear(input_dim, hidden_neurons, bias=True)
            hidden_layer  = nn.Linear(hidden_neurons, hidden_neurons, bias=True)
            output_layer  = nn.Linear(hidden_neurons, output_dim)
            activation    = nn.Sigmoid()
            dropout       = nn.Dropout(p_drop)

        network = nn.Sequential(input_layer,
                                activation,
                                dropout,
                                hidden_layer,
                                activation,
                                dropout,
                                output_layer) 
        return network


    def fit(self, X_train, Y_train, verbose=True, print_training_curve=False):
        """ Trains a one-layer neural network with modular input data dimension. """

        torch.manual_seed(42)

        ### Load hyperparameters
        var_prediction = self.hyperparams['var_prediction']
        self_normalize = self.hyperparams['self_normalize']
        hidden_neurons = self.hyperparams['hidden_neurons']
        p_drop         = self.hyperparams['dropout']
        batch_size     = self.hyperparams['batch_size']
        learning_rate  = self.hyperparams['learning_rate']
        epochs         = self.hyperparams['epochs']
        epsilon        = self.hyperparams['epsilon_pred']

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
        network = self.network_architecture(n_dim, hidden_neurons, var_prediction, p_drop, self_normalize)
        network.to(self.device)

        ### loss function
        if var_prediction:
            loss_function = self.neg_log_likelihood
        else:
            loss_function = nn.MSELoss() #nn.L1Loss() #nn.HuberLoss()

        ### Model training
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        adapt = True
        losses = []
        if verbose:
            print("Training neural network...")
        for epoch in tqdm(range(epochs)):
            if epoch > 1000 and adapt:
                optimizer = optim.Adam(network.parameters(), lr=learning_rate/10)
                adapt=False
            loss = 0
            for x, y in train_loader:
                x = torch.unsqueeze(x, dim=1)
                optimizer.zero_grad()

                if var_prediction:
                    mu_pred    = torch.unsqueeze((1 + epsilon) * torch.sigmoid(network(x)[:,0,0]) - epsilon/2, dim=1)
                    sigma_pred = torch.unsqueeze(network(x)[:,0,1], dim=1)
                    y_pred = torch.cat((mu_pred, sigma_pred), dim=1)
                else:
                    y_pred = (1 + epsilon) * torch.sigmoid(network(x)) - epsilon/2
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
            print(f"Final loss value: {losses[-1]}")

        if print_training_curve:
            plt.figure()
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Train Loss')
            plt.show()

    
    def predict(self, X, return_var=False):
        """ Computes Neural Network Output for given Input. """

        torch.manual_seed(42)

        self.network.eval()
        epsilon = self.hyperparams['epsilon_pred']
        var_prediction = self.hyperparams['var_prediction']

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        
        X = tensor(X, device=self.device).float()
        if var_prediction:
            Y_pred    = (1 + epsilon) * torch.sigmoid(self.network(X)[:,0]) - epsilon/2
            Sigma     = self.network(X)[:,1]
            if return_var:
                return Y_pred.cpu().detach().numpy(), Sigma.cpu().detach().numpy()
        else:
            Y_pred =  (1 + epsilon) * torch.sigmoid(self.network(X)) - epsilon/2
        return Y_pred.cpu().detach().numpy()

    
    def score(self, X, Y):
        """ Computes R2-score for given ground truth. """

        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)

    
    def neg_log_likelihood(self, y, y_pred):
        eps = self.hyperparams['epsilon_loss']
        mu_pred = y_pred[:,0]
        sigma_pred = y_pred[:,1]

        nll = torch.log(eps + torch.sqrt(2*3.1415927410125732*torch.square(sigma_pred))) + torch.div(torch.square(mu_pred-y), eps + 2*torch.square(sigma_pred))
        return nll.sum()



