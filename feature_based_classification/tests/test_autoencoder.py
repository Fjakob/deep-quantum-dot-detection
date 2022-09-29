from unittest import TestCase
import torch
import numpy as np
from src.lib.dimensionReduction.autoencoder import Autoencoder


class TestAutoencoder(TestCase):
    """ Test Suite Autoencoder """

    @classmethod
    def setUpClass(cls):
        """ Instanciate Autoencoder and Device """
        global autoencoder 
        global device

        autoencoder = Autoencoder(latent_dim=12)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def compare_tensor_dims(self, tensor1, tensor2):
        """ Compare two tensor sizes """
        equality = [size1 == tensor2.size()[idx] for idx, size1 in enumerate(tensor1.size())]
        return all(equality)


    ##########################################################################################
    #  T E S T    C A S E S
    ##########################################################################################

    def test_dimensions(self):
        """ Tests latent and output dimension of Autoencoder call. """
        input = torch.rand(4,1,1024, device=device)
        latent = autoencoder.encoder(input)
        output = autoencoder(input)

        self.assertTrue(self.compare_tensor_dims(latent, torch.randn(4,12)))
        self.assertTrue(self.compare_tensor_dims(input, output))


    def test_load_model(self):
        
        model_path = 'tests/fixtures/autoencoder12.pth'
        model_path_dim_error = 'tests/fixtures/autoencoder24.pth'

        autoencoder.load_model(model_path)  
        self.assertRaises(ValueError, autoencoder.load_model, model_path_dim_error)


    def test_reduce_raw(self):
        """ Tests the Autoencoder processing of raw datasets. """
        X = np.random.randn(4,1024)
        X_norm, Z, X_rec = autoencoder.normalize_and_reduce(X)

        self.assertIsInstance(X_rec, np.ndarray)
        self.assertTrue(X_norm.shape[0] == X_rec.shape[0] and X_norm.shape[1] == X_rec.shape[1])
        self.assertTrue((np.abs(X_norm) <= 1.0).all())
        self.assertTrue((np.abs(X_rec) <= 1.0).all())
