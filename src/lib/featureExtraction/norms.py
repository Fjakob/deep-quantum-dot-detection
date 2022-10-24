import numpy as np

def L2_norm(X1, X2):
    """ ... """

    l2 = np.linalg.norm(X1, X2, axis=1)
    return l2


def window_loss(X1, X2, window_size=9):
    """ ... """
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


def signal_window(x, idx, shift):
    """ ... """
    idx_left = idx - shift
    idx_right = idx + shift + 1

    if idx_left < 0:
        padding_left = np.zeros( abs(idx_left) )
        x_window = np.concatenate( (padding_left, x[0:idx_right]) )
    elif idx_right > len(x):
        padding_right = np.zeros( idx_right - len(x) )
        x_window = np.concatenate( (x[idx_left:len(x)], padding_right) )
    else:
        x_window = x[idx_left:idx_right]

    return x_window

