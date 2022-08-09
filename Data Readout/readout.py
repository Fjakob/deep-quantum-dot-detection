import os
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('LabeledSpectra_FJakob_v2.txt') as f:
    top_dir = "..\\..\\04_Daten\\Maps_for_ISYS\\"
    lines = f.readlines()
    dataSet = []
    for line in lines:
        line = line.split()
        labels = np.asarray(line[1:6]).astype(float)
        date = line[7]
        user = line[9]
        path = top_dir + line[-1] + ".dat"
        with open(path) as f:
            lines = f.readlines()
            w_raw = [line.split()[0] for line in lines]
            w = np.asarray(w_raw).astype(float)
            # read spectrum
            spectrum_raw = [line.split()[1] for line in lines]
            spectrum = np.asarray(spectrum_raw).astype(float)
        dataSet.append((w, spectrum, labels))


with open("dataSet", "wb") as fp:
    pickle.dump(dataSet, fp)

idx = 4
w   = dataSet[idx][0]
spec = dataSet[idx][1]
label = dataSet[idx][2]
plt.plot(w,spec)
plt.title(str(label[0]) + " peaks, " + str(label[1]) + " impression, " + str(label[3]) + " width")
plt.show()

