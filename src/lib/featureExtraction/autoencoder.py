import numpy as np
from torch import nn, cuda, tensor, load
from torchsummary import summary

from src.lib.featureExtraction.latent_extractor import LatentExtracter
from src.lib.neuralNetworks.encoder import ResidualEncoder as Encoder
from src.lib.neuralNetworks.decoder import DeepDecoder as Decoder


class Autoencoder(nn.Module, LatentExtracter):
    """ Autoencoder for dimensionality reduction. """
    def __init__(self, latent_dim=12, epsilon=1e-12):
        if latent_dim <= 0:
            raise ValueError("Latent dimension must be positive number")

        super(Autoencoder, self).__init__()
        device = "cuda" if cuda.is_available() else "cpu"

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, epsilon)

        self.device = device
        self.to(device)


    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded


    def load_model(self, model_path, model_summary=False):
        """ Loads neural network weights and biases from saved pretrained model. """
        try:
            self.load_state_dict(load(model_path))
        except(RuntimeError):
            raise ValueError("Loaded model doesn't fit object latent dimension.")

        #self.eval()
        
        if model_summary:
            summary(self, (1,1024))


    def extract_latent(self, X_normalized, return_reconstruction=False):
        # normalize and convert to torch
        X = tensor(X_normalized, device=self.device).float()
        X = X.view(X.shape[0], 1, X.shape[1])

        # encode and convert to numpy
        Z = self.encoder(X)

        if return_reconstruction:
            X_recon = self.forward(X)
            X_recon = np.squeeze(X_recon.cpu().detach().numpy())
            Z = Z.cpu().detach().numpy()
            return Z, X_recon

        Z = Z.cpu().detach().numpy()
        return Z

