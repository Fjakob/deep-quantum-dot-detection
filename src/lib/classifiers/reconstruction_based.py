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
        #e = np.linalg.norm(X_norm - X_hat, axis=1) #here shape loss
        #e = shape_loss(X_norm, X_hat)
        e = window_loss(X_norm, X_hat)
        e = torch.tensor(e, device=device).float()
        e = torch.unsqueeze(e, dim=1)
        y = self.regressor(e)
        y = torch.squeeze(y)
        y = y.cpu().detach().numpy()
        return y


def signal_window(x, idx, shift):
    idx_left = idx - shift
    idx_right = idx + shift + 1

    # take signal window
    if idx_left < 0:
        padding_left = np.zeros( abs(idx_left) )
        x_window = np.concatenate( (padding_left, x[0:idx_right]) )
    elif idx_right > len(x):
        padding_right = np.zeros( idx_right - len(x) )
        x_window = np.concatenate( (x[idx_left:len(x)], padding_right) )
    else:
        x_window = x[idx_left:idx_right]

    return x_window


def window_loss(X1, X2, window_size=9):
    if len(X1.shape) == 1:
        X1 = np.expand_dims(X1, axis=0)
    if len(X2.shape) == 1:
        X2 = np.expand_dims(X2, axis=0)
    n = X1.shape[0]
    shift = int((window_size-1)/2)

    loss = []
    idx=0
    for idx in range(n):
        x1 = X1[idx]
        x2 = X2[idx]
        diff = x1-x2
        e = []
        for idx in range(len(diff)):
            window = signal_window(diff, idx, shift)
            e.append(np.mean(window))
        loss.append(np.linalg.norm(e))
    return np.asarray(loss)
