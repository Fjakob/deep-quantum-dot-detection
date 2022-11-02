from config.imports import *
import torch

from src.lib.featureExtraction.autoencoder import Autoencoder


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


def shape_loss(x1, x2, window_size=3, exploration_space=2):
    x1 = np.squeeze(x1)
    x2 = np.squeeze(x2)
    n = x1.shape[0]

    shift = int((window_size-1)/2)

    errors = []
    for idx in range(n):
        local_errors = []
        x1_window = signal_window(x1, idx, shift)
        x2_window = signal_window(x2, idx, shift)
        local_errors.append(np.linalg.norm(x1_window-x2_window)/len(x1_window))
        for jdx in range(1, exploration_space+1):
            x2_window_up = signal_window(x2, idx+jdx, shift)
            x2_window_down = signal_window(x2, idx-jdx, shift)
            local_errors.append(np.linalg.norm(x1_window-x2_window_up)/len(x1_window))
            local_errors.append(np.linalg.norm(x1_window-x2_window_down)/len(x2_window))
        e = np.min(local_errors)
        errors.append(e)
    return np.linalg.norm(errors)

def window_loss(x1, x2, window_size=11):
    x1 = np.squeeze(x1)
    x2 = np.squeeze(x2)
    diff = x1 - x2
    shift = int((window_size-1)/2)

    e = []
    for idx in range(len(diff)):
        window = signal_window(diff, idx, shift)
        e.append(np.mean(window))

    return np.linalg.norm(e)


def main():

    errors = []

    #latent_dims = [12, 16, 24, 32, 64, 128, 256]
    latent_dims = [32]

    for latent_dim in latent_dims:
        shift = 200
        
        autoencoder = Autoencoder(latent_dim=latent_dim)
        autoencoder.load_model(f'models/autoencoders/autoencoder{latent_dim}.pth')

        with open('datasets\labeled\data_w30_labeled.pickle', 'rb') as file:
            X, Y = pickle.load(file)

        print(f"Latent dim {latent_dim}")
        recon_errors = []
        recon_errors_shape = []
        idx=0
        for x in X:
            x_1, z_1, x_r1 = autoencoder.normalize_and_extract(x)
            x = np.roll(x, shift=shift)
            x_2, z_2, x_r2 = autoencoder.normalize_and_extract(x)

            print(f"Norm between x1 and x2: {np.linalg.norm(x_1-x_2)}")
            print(f"Norm between z1 and z2: {np.linalg.norm(z_1-z_2)}")
            print(f"Norm between x1-x_r1 and x2-x_r2: {np.linalg.norm(x_1-x_r1)} and {np.linalg.norm(x_2-x_r2)}")
            print(f"window_loss between x1-x_r1 and x2-x_r2: {window_loss(x_1, x_r1)} and {window_loss(x_2, x_r2)}")
            print('--------------------')

            if np.linalg.norm(x_1-x_r1) < 5:
                recon_errors.append(np.linalg.norm(x_1-x_r1))
                recon_errors.append(np.linalg.norm(x_2-x_r2))
                recon_errors_shape.append(window_loss(x_1, x_r1))
                recon_errors_shape.append(window_loss(x_2, x_r2))

            plt.figure(figsize=(10,6))
            plt.suptitle(f'Label: {Y[idx]}')
            plt.subplot(3,1,1)
            plt.plot(x_1[0])
            plt.plot(x_r1, '--')
            plt.subplot(3,1,2)
            plt.plot(x_2[0])
            plt.plot(x_r2, '--')
            plt.subplot(3,1,3)
            plt.plot(z_1[0])
            plt.plot(z_2[0])
            plt.show()

            idx+=1
            
        print(f"Mean loss: {np.mean(recon_errors)}")
        print(f"Mean shape loss: {np.mean(recon_errors_shape)}\n")

        errors.append([latent_dim, np.mean(recon_errors), np.mean(recon_errors_shape)])

    errors = np.asarray(errors)

    plt.figure()
    plt.xscale("log")
    plt.plot(errors[:,0], errors[:,1], label="MSE")
    plt.plot(errors[:,0], errors[:,2], label="Shape loss")
    plt.xlabel('Latent dimension')
    plt.show()
        

if __name__ == "__main__":
    main()

