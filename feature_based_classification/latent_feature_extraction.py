import torch
from torch import nn 
from torchsummary import summary
from neuralNetworks.autoencoder import autoencoder


def load_autoencoder(latent_dim=12):
    """ Loads saved model into a new autoencoder instance. """

    model_path = 'autoencoders/autoencoder{}.pth'.format(latent_dim)

    model_autoencoder = autoencoder(latent_dim).to('cuda')
    model_autoencoder.load_state_dict(torch.load(model_path))

    summary(model_autoencoder, (1,1024))
    return model_autoencoder


if __name__ == '__main__':

    load_autoencoder(latent_dim=12)


