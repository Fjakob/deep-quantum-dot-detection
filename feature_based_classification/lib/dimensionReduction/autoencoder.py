import numpy as np
from torch import nn, tensor
from lib.neuralNetworks.encoder import residualEncoder as encoder
from lib.neuralNetworks.decoder import decoder_deep as decoder

class autoencoder(nn.Module):
    """ Autoencoder for dimensionality reduction. """
    def __init__(self, latent_dim=12):
        super(autoencoder, self).__init__()
        self.encoder = encoder(latent_dim)
        self.decoder = decoder(latent_dim)
        
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

    def reduce_raw(self, X):
        """ 
        Takes raw unnormalized dataset in numpy data format;
        returns normalized data, reconstruction and latent respresentation in numpy data format. 
        """
        X_scaled = X / np.max(np.abs(X), axis=1)[:,np.newaxis]

        X = tensor(X_scaled, device='cuda').float()
        X = X.view(X.shape[0], 1, X.shape[1])
        Z = self.encoder(X)
        X_hat = self.decoder(Z)

        # convert from torch to np
        Z = Z.cpu().detach().numpy()
        X = X.cpu().detach().numpy()
        X_hat = X_hat.cpu().detach().numpy()

        return np.squeeze(X), Z, np.squeeze(X_hat)


