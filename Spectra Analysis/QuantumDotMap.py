import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def readFiles(dir):
    # extract all .DAT files:
    files = glob.glob(dir + '/*.' + 'DAT')
    print('\nFound {0} files (Arising from {1}x{1} grid)\n'.format(len(files), int(np.sqrt(len(files)))))

    # Read files and save into dictionary
    spectra = dict()
    FIRST_FILE = True
    for file in files:
        with open(file) as f:
            # remove file ending and extract infos from file name for spatial coordinates
            file_info = ''.join(os.path.basename(file).split('.')[:-1]).split("_")
            x_coord = float(file_info[1].lstrip('X'))/1000
            y_coord = float(file_info[2].lstrip('Y'))/1000
            # read spectrum
            lines = f.readlines()
            spectrum = [line.split()[1] for line in lines]
            spectra[(x_coord, y_coord)] = np.asarray(spectrum).astype(float)
            if FIRST_FILE:
                # read wavelength (only once, since constant)
                w_raw = [line.split()[0] for line in lines]
                w = np.asarray(w_raw).astype(float)
                FIRST_FILE = False
    return w, spectra


if __name__ == '__main__':

    # set directories
    pwd = os.getcwd()
    dir = pwd + '\\..\\..\\04_Daten\\sample'
    os.chdir(dir)

    # extract spectras
    wavelengths, spectras = readFiles(dir)

    # plot grid with maximum spectrum peak as colormap
    x = y = np.linspace(-3000,3000,31)
    z = np.array([np.max(spectras[(i,j)]) for j in y for i in x])
    z = z.reshape(31,31)

    plt.figure()
    plt.imshow(z[::-1], cmap='bwr', interpolation='none', extent=[-3000, 3000, -3000, 3000])
    plt.show()
