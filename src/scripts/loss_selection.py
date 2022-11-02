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


def window_loss(x1, x2, window_size=3):
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

    latent_dims = [8, 16, 24, 32, 48, 64]

    short_window = 3
    large_window = 9

    for latent_dim in latent_dims:
        
        autoencoder = Autoencoder(latent_dim=latent_dim)
        autoencoder.load_model(f'models/autoencoders/epsilon_1e-12/autoencoder{latent_dim}.pth')

        with open('datasets\labeled\data_w30_labeled.pickle', 'rb') as file:
            X, Y = pickle.load(file)

        print(f"Latent dim {latent_dim}")
        recon_errors = []
        recon_errors_short = []
        recon_errors_large = []
        idx=0
        for x in X:
            x_1, z_1, x_r1 = autoencoder.normalize_and_extract(x)
            #x = np.roll(x, shift=shift)
            #x_2, z_2, x_r2 = autoencoder.normalize_and_extract(x)

            # print(f"Norm between x1 and x2: {np.linalg.norm(x_1-x_2)}")
            # print(f"Norm between z1 and z2: {np.linalg.norm(z_1-z_2)}")
            # print(f"Norm between x1-x_r1 and x2-x_r2: {np.linalg.norm(x_1-x_r1)} and {np.linalg.norm(x_2-x_r2)}")
            # print(f"window_loss between x1-x_r1 and x2-x_r2: {window_loss(x_1, x_r1)} and {window_loss(x_2, x_r2)}")
            # print('--------------------')

            if np.linalg.norm(x_1-x_r1) < 5:
                recon_errors.append(np.linalg.norm(x_1-x_r1))
                recon_errors_short.append(window_loss(x_1, x_r1, window_size=short_window))
                recon_errors_large.append(window_loss(x_1, x_r1, window_size=large_window))

            # plt.figure(figsize=(10,6))
            # plt.suptitle(f'Label: {Y[idx]}')
            # plt.subplot(3,1,1)
            # plt.plot(x_1[0])
            # plt.plot(x_r1, '--')
            # plt.subplot(3,1,2)
            # plt.plot(x_2[0])
            # plt.plot(x_r2, '--')
            # plt.subplot(3,1,3)
            # plt.plot(z_1[0])
            # plt.plot(z_2[0])
            # plt.show()

            idx+=1
            
        print(f"Mean loss: {np.mean(recon_errors)}")
        print(f"Mean {short_window}-windowed loss: {np.mean(recon_errors_short)}")
        print(f"Mean {large_window}-windowed loss: {np.mean(recon_errors_large)}\n")

        errors.append([latent_dim, np.mean(recon_errors), np.mean(recon_errors_short), np.mean(recon_errors_large)])

    errors = np.asarray(errors)

    plt.figure()
    #plt.xscale("log")
    plt.plot(errors[:,0], errors[:,1], '-*r', label="L2 norm")
    plt.plot(errors[:,0], errors[:,2], '-*b', label=f"Windowed L2 norm (w={short_window})")
    plt.plot(errors[:,0], errors[:,3], '-*g', label=f"Windowed L2 norm (w={large_window})")
    plt.xlabel('Latent dimension')
    plt.ylabel('Mean reconstruction error')
    plt.grid()
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    main()

