import os
from os.path import isfile
import glob
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class DataProcesser():
    """ Data Loader class for data base readout of IHFG quantum dots. """
    def __init__(self, absolute_database_path, spectrum_size=1024):
        self.db_path = absolute_database_path
        self.spectrum_size = spectrum_size


    def load_spectra_from_database(self, saving_path=None):
        """ Reads all spectra in all folders of a given database """
        spectra = dict()
        folders = os.listdir(self.db_path)
        for idx, folder in enumerate(folders):
            print(f"Folder {idx+1} of {len(folders)}")
            dir = os.path.join(self.db_path, folder)
            new_spectra = self.load_spectra_from_dir(dir)
            spectra = merge_dictionary_contents(spectra, new_spectra)

        if saving_path is not None:
            self.save(spectra, saving_path, file_name='database')
        else:
            return spectra
    

    def load_spectra_from_dir(self, dir=None, folder_nmb=1):
        """ 
        Loads all spectra of a diven directory containing .DAT CCD files.
        If no directory is given, a database folder will be selected as database, specified by folder_nmb.
        Optional noise or background filtering adjustable.
        """
        if dir is None:
            folders = next(os.walk(self.db_path))[1]
            if folder_nmb > len(folders) or folder_nmb < 1:
                raise ValueError("Unvalid folder number.")
            dir = os.path.join(self.db_path, folders[folder_nmb-1])

        files = glob.glob(dir + '/*.' + 'DAT')
        spectra = dict()
        for file in tqdm(files):
            with open(file) as f:
                lines = f.readlines()

                # assert data length
                if len(lines) != self.spectrum_size:
                    error_string = f"Data length of file {file} in dir {dir} inconsistent (expected {self.spectrum_size} samples)"
                    raise DataLengthError(error_string)

                wavelengths  = np.asarray([line.split()[0] for line in lines]).astype(float)
                spectrum = np.asarray([line.split()[1] for line in lines]).astype(float)
                
                w_range = int(np.floor(wavelengths[-1] - wavelengths[0]))
                
                if f"{w_range}" not in spectra.keys():
                    spectra[f"{w_range}"] = []
                spectra[f"{w_range}"].append(spectrum)     

        for key in spectra.keys():
            spectra[key] = np.asarray(spectra[key])   
        return spectra

    
    def load_spectra_from_file(self, file_path):

        if isfile(file_path):
            with open(file_path, 'rb') as f:
                database = pickle.load(f)
            return database
        else:
            raise FileNotFoundError


    def create_unsupervised_data(self, w_range, database=None, loading_path=None, saving_path=None, 
                    normalize=True, augment=False, space_shifts=2, mirroring=False,
                    noise_filtering=True, noise_bound=50, background_filtering=False, background_bound=60):
        """  
        1) Unify all spectra to same wavelength scope
        2) Filter out noise
        3) Filter out background
        4) normalize
        5) augment
        6) return as np.ndarray
        dataset: python.dict -> np.ndarray
        """

        if database is None:
            if loading_path is None:
                database = self.load_spectra_from_database()
            else:
                database = self.load_spectra_from_file(loading_path)

        file_name = f'data_w{w_range}_unlabeled'
        X = []
        count = 0
        for spectrum in database[f"{w_range}"]:
            # do not consider noisy or background containing spectra
            if noise_filtering and np.max(spectrum) < noise_bound:
                count += 1 
                continue
            if background_filtering and np.max(spectrum[-200:]) > background_bound:
                count += 1
                continue
            X.append(list(spectrum))

        X = np.asarray(X)
        print(f"Filtered out {count} noisy spectra")
        print(f"Dataset size: {X.shape[0]}")

        if normalize:
            X = X / np.max(np.abs(X), axis=1)[:,np.newaxis]
            file_name += '_normalized'

        if augment:
            X = self.augment(X, space_shifts=space_shifts, mirroring=mirroring)
            file_name += '_augmented'

        if saving_path is not None:
            self.save(X, saving_path, file_name)
            os.system(f'echo {file_name}.pickle >> {saving_path}//.gitignore')
        else:
            return X       


    def create_regression_data(self, w_range, txt_dir, return_peak_count=False,
                                show_statistics=True, saving_path=None,
                                augment=False, space_shifts=5, mirroring=True):

        ################## READOUT: #####################
        userSet, spectra_dict = dict(), dict()
        label_dir = txt_dir + f"\\w{w_range}"
        for file in os.listdir(label_dir):
            labelDict = dict()
            with open(os.path.join(label_dir, file)) as f:
                user = file.split("_")[1]
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    labels = np.asarray(line[1:3]).astype(float)
                    if labels[1] == -2 or labels[1] == -1:
                        labels[1] += 2
                    elif labels[1] == 1 or labels[1] == 2:
                        labels[1] += 1
                    file_name = line[-1].split("\\")[1]
                    path = f"{self.db_path}\\{line[-1]}.dat"
                    with open(path) as f:
                        lines = f.readlines()
                        spectrum_raw = [line.split()[1] for line in lines]
                        spectrum = np.asarray(spectrum_raw).astype(float)
                        labelDict[file_name] = labels
                        if file_name not in spectra_dict:
                            spectra_dict[file_name] = spectrum
            userSet[user] = labelDict

        ################## EVALUATION: #####################
        X, Y, P = [], [], []
        for file_name in spectra_dict:
            x, labels, peak_counts = spectra_dict[file_name], [], []
            for user, user_labels in userSet.items():
                if file_name in user_labels.keys():
                    peak_count, label = user_labels[file_name] 
                    labels.append(label)
                    peak_counts.append(peak_count)
            y = np.mean(labels) / 3
            peaks = round(np.mean(peak_counts))
            X.append(x), Y.append(y), P.append(peaks)
        X, Y = np.asarray(X), np.asarray(Y)
        file_name = f'data_w{w_range}_labeled'

        if augment:
            X, Y = self.augment(X, Y, space_shifts=space_shifts, mirroring=mirroring)
            file_name += '_augmented'
        
        print(f"Created {Y.shape[0]} regression data points.")

        if show_statistics:
            plt.hist(Y, align='mid', bins=50)
            plt.title('Distribution of labels')
            plt.show()
        
        if saving_path is not None:
            if not return_peak_count:
                self.save((X,Y), saving_path, file_name)
            else:
                file_name += '_with_peaks'
                self.save((X,Y,P), saving_path, file_name )
            os.system(f'echo {file_name}.pickle >> {saving_path}//.gitignore')
        else:
            if not return_peak_count:
                return X, Y
            else:
                return X, Y, P

    
    def augment(self, X, Y=None, space_shifts=2, mirroring=True, saving_path=None):

        space_shifts+=1
        if space_shifts > self.spectrum_size:
            raise ValueError(f"Number of space shifts must be smaller than array size {self.spectrum_size}.")
        shift = round(self.spectrum_size/space_shifts)
        print(f"Increase data size by factor {space_shifts*(2 if mirroring else 1)}")

        X_augmented = []
        if Y is None:
            for x in X:
                for jdx in range(1, space_shifts+1):
                    x_shifted = np.roll(x, jdx*shift)
                    X_augmented.append(x_shifted)
                    if mirroring:
                        x_mirrored = np.flip(x_shifted)
                        X_augmented.append(x_mirrored)
            X_augmented = np.asarray(X_augmented)

            if saving_path is not None:
                self.save(X_augmented, saving_path, 'data_unlabeled_augmented')
            return X_augmented

        if Y is not None:
            Y_augmented = []
            idx = 0
            for x in X:
                y = Y[idx]
                idx += 1
                for jdx in range(1, space_shifts+1):
                    x_shifted = np.roll(x, jdx*shift)
                    X_augmented.append(x_shifted)
                    Y_augmented.append(y)
                    if mirroring:
                        x_mirrored = np.flip(x_shifted)
                        X_augmented.append(x_mirrored)
                        Y_augmented.append(y)
            if saving_path is not None:
                self.save(X, saving_path, 'data_labeled_augmented')
            return np.asarray(X_augmented), np.asarray(Y_augmented) 


    def save(self, obj, saving_path, file_name):
        with open("".join([saving_path, f"\{file_name}.pickle"]), 'wb') as f:
            pickle.dump(obj, f)
            print(f"Saved data as {file_name}.pickle")
    

    def create_artificial_spectra(self, n_samples, seed=42, noise_var=2.5, max_peak_height=100):
        """ Creates n artificial spectra with one peak and increasing SNR. """
        np.random.seed(seed)
        min_peak_height=10
        X = []
        for idx in range(n_samples):
            peak_height = min_peak_height + (idx+1)/n_samples * (max_peak_height-min_peak_height)
            spectrum = np.random.normal(0.25, noise_var, size=self.spectrum_size)
            spectrum[int(self.spectrum_size/2)] = peak_height
            spectrum[int(self.spectrum_size/2)-1] = peak_height/2
            spectrum[int(self.spectrum_size/2)+1] = peak_height/2
            X.append(spectrum)
        return np.asarray(X)


class DataLengthError(Exception):
    """ An exception that is raised when a spectrum has not the specified number of samples. """
    pass

def merge_dictionary_contents(dict1, dict2):
    merged_dict = dict1
    for key, value in dict2.items():
        if key in merged_dict.keys():
            merged_dict[key] = np.vstack((merged_dict[key], value))
        else:
            merged_dict[key] = value
    return merged_dict
            