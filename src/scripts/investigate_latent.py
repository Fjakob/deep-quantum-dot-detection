from config.imports import *

from src.lib.featureExtraction.autoencoder import Autoencoder


def main():

    latent_dim = 128
    shift = 100
    
    autoencoder = Autoencoder(latent_dim=latent_dim)
    autoencoder.load_model(f'models/autoencoders/autoencoder{latent_dim}.pth')

    with open('datasets\labeled\data_w30_labeled.pickle', 'rb') as file:
        X, _ = pickle.load(file)
    np.random.shuffle(X)

    for x in X:
        x_1, z_1, x_r1 = autoencoder.normalize_and_extract(x)
        x = np.roll(x, shift=shift)
        x_2, z_2, x_r2 = autoencoder.normalize_and_extract(x)
        print(f"Norm between x1 and x2: {np.linalg.norm(x_1-x_2)}")
        print(f"Norm between z1 and z2: {np.linalg.norm(z_1-z_2)}")
        print(f"Norm between x1-x_r1 and x2-x_r2: {np.linalg.norm(x_1-x_r1)} and {np.linalg.norm(x_2-x_r2)}")
        print('--------------------')
        plt.figure(figsize=(10,6))
        plt.suptitle(f'Space shift = {shift}')
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
        

if __name__ == "__main__":
    main()