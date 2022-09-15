from torch import nn
from neuralNetworks.encoder import residualEncoder as encoder
from neuralNetworks.decoder import decoder

class autoencoder(nn.Module):
    """ Autoencoder for dimensionality reduction. """
    def __init__(self, latent_dim=32):
        super(autoencoder, self).__init__()
        self.encoder = encoder(latent_dim)
        self.decoder = decoder(latent_dim)
        
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

