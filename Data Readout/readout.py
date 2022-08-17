import os
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import pickle


def loadDataSet(label_dir, data_dir):
    """ Loads all labeled spectra from txt files."""
    dataSet = []
    for file in os.listdir(label_dir):
        with open(label_dir + '\\' + file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                labels = np.asarray(line[1:6]).astype(float)
                #date = line[7]
                #user = line[9]
                path = data_dir + line[-1] + ".dat"
                try:
                    with open(path) as f:
                        lines = f.readlines()
                        w_raw = [line.split()[0] for line in lines]
                        w = np.asarray(w_raw).astype(float)
                        # read spectrum
                        spectrum_raw = [line.split()[1] for line in lines]
                        spectrum = np.asarray(spectrum_raw).astype(float)
                except(FileNotFoundError):
                    pass
                dataSet.append((w, spectrum, labels))
    return dataSet


if __name__ == '__main__':

    plot = False

    # Specify working directories
    label_dir = 'LabeledSpectra'
    data_dir = "..\\..\\04_Daten\\Maps_for_ISYS\\"

    # Load dataset
    dataSet = loadDataSet(label_dir, data_dir)

    # save as pickle
    with open("dataSet", "wb") as fp:
        pickle.dump(dataSet, fp)
    print("Saved data set with {0} labeled spectra.".format(len(dataSet)))

    if plot:
        for w, spec, label in dataSet:
            plt.plot(w,spec)
            plt.title(str(label[0]) + " peaks, " 
                        + str(label[1]) + " impression, " 
                        + str(label[2]) + " background\n" 
                        + str(label[3]) + " distinctness, " 
                        + str(label[3]) + " width")
            plt.show()

