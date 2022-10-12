import numpy as np
import torch

class ReconstructionBasedRater():
    """ Class for spectrum rating """
    ######################################################
    # TODO: make modular for different types of regressors
    ######################################################
    def __init__(self, reconstructer, regressor):
        self.reconstructer  = reconstructer
        self.regressor = regressor

    def rate(self, X):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_norm, _, X_hat = self.reconstructer.normalize_and_extract(X)
        e = np.linalg.norm(X_norm - X_hat, axis=1)
        e = torch.tensor(e, device=device)
        e = torch.unsqueeze(e, dim=1)
        y = self.network(e)
        y = torch.squeeze(y)
        y = y.cpu().detach().numpy()
        return y

