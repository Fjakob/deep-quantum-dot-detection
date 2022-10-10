import os
from os.path import isfile
import glob
import pickle
import numpy as np
from tqdm import tqdm


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


    def save(self, obj, saving_path, file_name):
        with open("".join([saving_path, f"\{file_name}.pickle"]), 'wb') as f:
            pickle.dump(obj, f)
            print(f"Saved data as {file_name}.pickle")


    def preprocess(self, database, w_range, saving_path=None, normalize=True,
                    noise_filtering=True, noise_bound=50, background_filtering=False, background_bound=60):
        """  
        1) Unify all spectra to same wavelength scope
        2) Filter out noise
        3) Filter out background
        4) normalize
        5) return as np.ndarray
        dataset: python.dict -> np.ndarray
        """
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

        if saving_path is not None:
            self.save(X, saving_path, 'data_unlabeled_normalized')
        return X


    def augment(self, X, Y=None, space_shifts=3, mirroring=True, saving_path=None):

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


    def print_data_statistic():
        """ 
        Plot sample data
        Plot mean spectrum
        Yield mean, std, etc
        """
        pass


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