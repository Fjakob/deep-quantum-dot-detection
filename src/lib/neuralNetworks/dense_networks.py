import numpy as np
import torch
from torch import nn, tensor, cuda, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm


class VanillaNeuralNetwork(nn.Module):
    """ One Layer Network with Sigmoid activation function. """
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.device = "cuda" if cuda.is_available() else 'cpu'
        self.network = None


    def fit(self, X, Y):
        """ Trains a one-layer neural network with modular input data dimension. """
        ### Load hyperparameters
        hidden_neurons = self.hyperparams['hidden_neurons']
        p_drop         = self.hyperparams['dropout']
        batch_size     = self.hyperparams['batch_size']
        learning_rate  = self.hyperparams['learning_rate']
        epochs         = self.hyperparams['epochs']

        ### Extract dimensions
        if X.shape == 1:
            X = np.expand_dims(X, axis=1)
        n_dim = X.shape[1]

        ### Data preparation
        X = tensor(X, device=self.device)
        Y = tensor(Y, device=self.device)
        dataset = TensorDataset(X.float(), Y.float())
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        ### Model initialization
        network = nn.Sequential(nn.Linear(n_dim, hidden_neurons), 
                                nn.Sigmoid(), 
                                nn.Dropout(p_drop), 
                                nn.Linear(hidden_neurons, 1),
                                nn.Sigmoid())
        network.to(self.device)

        ### Model training
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss() #nn.L1Loss() #nn.HuberLoss()
        losses = []
        print("Training neural network...")
        for epoch in tqdm(range(epochs)):
            loss = 0
            for x, y in train_loader:
                x = torch.unsqueeze(x, dim=1)
                optimizer.zero_grad() 
                y_pred = network(x)
                y_pred = torch.squeeze(y_pred)
                train_loss = loss_function(y, y_pred)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(train_loader)
            losses.append(loss)

        R2 = r2_score(Y.cpu().detach().numpy(), network(X.float()).cpu().detach().numpy())
        print(f'Training R2-score: {R2}')

        """     plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.show() """
        
        self.network = network


    def predict(self, X):
        if X.shape == 1:
            X = np.expand_dims(X, axis=1)
        X = tensor(X, device=self.device).float()
        Y_pred = self.network(X)
        return Y_pred.cpu().detach().numpy()
         
