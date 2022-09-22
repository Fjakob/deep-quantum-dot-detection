import torch
import numpy as np
from src.lib.dimensionReduction.autoencoder import autoencoder

class TestAutoencoder():

    """ Helper functions. """
    def compare_tensor_sizes(self, tensor1, tensor2):
        equality = [size1 == tensor2.size()[idx] for idx, size1 in enumerate(tensor1.size())]
        return all(equality)


    """ Test functions. """
    def test_instanciation(self):
        """ Tests the instanciation of the autoencoder class. """
        ae = autoencoder()
        assert isinstance(ae, autoencoder)


    def test_dimensions(self):
        """ Tests latent and output dimension of autoencoder call. """
        ae = autoencoder(latent_dim=12)

        input = torch.rand(4,1,1024)
        latent = ae.encoder(input)
        output = ae(input)

        assert self.compare_tensor_sizes(latent, torch.randn(4,12))
        assert self.compare_tensor_sizes(input, output)


    def test_cuda(self):
        """ Tests whether cuda is available on device. """
        assert torch.cuda.is_available()


    def test_reduce_raw(self):
        """ Tests the autoencoder processing of raw datasets. """
        ae = autoencoder(latent_dim=12).to('cuda')

        X = np.random.randn(4,1024)
        X_norm, Z, X_rec = ae.reduce_raw(X)

        assert isinstance(X_rec, np.ndarray)
        assert X_norm.shape[0] == X_rec.shape[0] and X_norm.shape[1] == X_rec.shape[1]
        assert (np.abs(X_norm) <= 1.0).all()
        assert (np.abs(X_rec) <= 1.0).all()