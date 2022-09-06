import os
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import pickle
    
    
def loadDataSet(label_dir, data_dir):
    """ Loads all labeled spectra from txt files, sorted by User."""
    userSet = dict()
    file_names = dict()
    for file in os.listdir(label_dir):
        labelDict = dict()
        with open(os.path.join(label_dir, file)) as f:
            user = file.split("_")[1]
            lines = f.readlines()
            for line in lines:
                line = line.split()
                assert user == line[6]
                labels = np.asarray(line[1:3]).astype(float)
                name = line[-1].split("\\")[1]
                path = os.path.join(data_dir, line[-1]+".dat")
                with open(path) as f:
                    lines = f.readlines()
                    w_raw = [line.split()[0] for line in lines]
                    w = np.asarray(w_raw).astype(float)
                    # read spectrum
                    spectrum_raw = [line.split()[1] for line in lines]
                    spectrum = np.asarray(spectrum_raw).astype(float)
                    labelDict[name] = labels
                    if name not in file_names:
                        file_names[name] = (w, spectrum)
        userSet[user] = labelDict
    return file_names, userSet


if __name__ == '__main__':

    plot_user_statistics = False
    plot_clustered = True
    
    top_dir = os.getcwd()

    # Specify working directories
    label_dir = 'LabeledSpectra_v3'
    data_dir = "Maps_for_ISYS\\"

    # Load dataset
    file_names, userSet = loadDataSet(label_dir, data_dir)

    if plot_user_statistics:
        for name in file_names:
            w, x = file_names[name]
            fig, ax = plt.subplots() 
            ax.plot(w,x)
            text = ""
            for user in userSet:
                try:
                    label = userSet[user][name]
                    text += (user + ": {0} peaks, {1} impression\n".format(label[0],label[1]))
                except(KeyError):
                    pass
            ax.text(0.0, 1, text, fontsize = 10, transform=ax.transAxes)
            plt.show()
    
    
    if plot_clustered:      
        expert = userSet["AB"]
        veryBad = [names for names, labels in expert.items() if labels[1]==-2]
        Bad = [names for names, labels in expert.items() if labels[1]==-1]
        Ok = [names for names, labels in expert.items() if labels[1]==1]
        Good = [names for names, labels in expert.items() if labels[1]==2]
        
        clusteredSpectra = {"-2": veryBad, "-1": Bad, "1": Ok, "2": Good}
        
        impressions = ["-2","-1","1","2"]
        
        for impression in impressions:
            idx = 0
            fig = plt.figure()
            for name in clusteredSpectra[impression]:
                idx+=1
                w, x = file_names[name]
                ax = fig.add_subplot(3,2,idx)
                ax.plot(w, x)
                if idx==6:
                    break
            fig.suptitle("Impression: " + impression)
            plt.show()