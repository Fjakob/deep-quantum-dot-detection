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

                ### assert data length
                if len(lines) != self.spectrum_size:
                    error_string = f"Data length of file {file} in dir {dir} inconsistent (expected {self.spectrum_size} samples)"
                    raise DataLengthError(error_string)

                ### retrieve information
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
        """ Returns content of given pickle file. """
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

        ### allocate database
        if database is None:
            if loading_path is None:
                database = self.load_spectra_from_database()
            else:
                database = self.load_spectra_from_file(loading_path)

        ### read out all files having w_range as wavelength range
        file_name = f'data_w{w_range}_unlabeled'
        X, count = [], 0
        for spectrum in database[f"{w_range}"]:
            
            ### do not consider noisy or background containing spectra
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

        ### normalize to value range [-1, 1]
        if normalize:
            X = X / np.max(np.abs(X), axis=1)[:,np.newaxis]
            file_name += '_normalized'

        ### augment data with given space shifts
        if augment:
            X = self.augment(X, space_shifts=space_shifts, mirroring=mirroring)
            file_name += '_augmented'

        ### save data and add to .gitignore
        if saving_path is not None:
            self.save(X, saving_path, file_name)
            os.system(f'echo {file_name}.pickle >> {saving_path}//.gitignore')
        else:
            return X       


    def create_regression_data(self, w_range, txt_dir, file_ending=None, return_peak_count=False,
                                show_statistics=True, saving_path=None,
                                augment=False, space_shifts=5, mirroring=True):
        """  
        1) Find all spectra with the wavelength range w_range
        2) Read out txt of labels
        3) Average the labels from all users, assigned to individual labels
        4) Optional: augment with space_shifts and mirroring
        """

        ################## TXT READOUT: #######################
        label_dir = txt_dir + f"\\w{w_range}"
        if file_ending is not None:
            label_dir += file_ending
        user_dictionary, spectrum_storage = dict(), dict()

        for file in os.listdir(label_dir):

            ### for each user, store the mapping file_name->label
            labeled_files_dictionary = dict() 
            with open(os.path.join(label_dir, file)) as f:
                user = file.split("_")[1]

                lines = f.readlines()
                for line in lines:
                    line = line.split()

                    ### ignore spectra having different wavelength range than w_range
                    if line[3] == 'w_range' and int(line[4]) is not w_range:
                        continue
                    
                    ### retrieve information from file
                    labels = np.asarray(line[1:3]).astype(float)
                    file_name = line[-1].split("\\")[1]
                    path = f"{self.db_path}\\{line[-1]}.dat"

                    ### if labels has been rated based on 4 classes, transform to [0, 1]
                    if labels[1] == -2 or labels[1] == -1:
                        labels[1] += 2
                        labels[1] = labels[1] / 3
                    elif labels[1] == 1 or labels[1] == 2:
                        labels[1] += 1
                        labels[1] = labels[1] / 3

                    labeled_files_dictionary[file_name] = labels

                    ### add to storage of all file_name->spectrum mappings, if not already contained
                    if file_name not in spectrum_storage:
                        with open(path) as f:
                            lines = f.readlines()
                            spectrum_raw = [line.split()[1] for line in lines]
                            spectrum = np.asarray(spectrum_raw).astype(float)
                            spectrum_storage[file_name] = spectrum

            user_dictionary[user] = labeled_files_dictionary


        ################## DATA SET CREATION: #####################
        X, Y, P = [], [], []
        
        ### for all stored spectrum, assign label as the mean of all users that rated it
        for file_name in spectrum_storage:
            x, labels, peak_counts = spectrum_storage[file_name], [], []
            
            ### check all users who rated the spectrum
            for user, user_labels in user_dictionary.items():
                if file_name in user_labels.keys():
                    peak_count, label = user_labels[file_name] 
                    labels.append(label)
                    peak_counts.append(peak_count)

            y = np.mean(labels)
            peaks = round(np.mean(peak_counts))
            X.append(x), Y.append(y), P.append(peaks)

        X, Y, P = np.asarray(X), np.asarray(Y), np.asarray(P)
        file_name = f'data_w{w_range}_labeled'

        ### augment data with given space shifts
        if augment:
            X, Y = self.augment(X, Y, space_shifts=space_shifts, mirroring=mirroring)
            file_name += '_augmented'

        ### plot histogram of label distribution
        if show_statistics:
            plt.hist(Y, align='mid', bins=50)
            plt.title('Distribution of labels')
            plt.show()

        print(f"Created {Y.shape[0]} regression data points.")
        
        ### save data or return directly
        if saving_path is not None:
            if file_ending is not None:
                file_name += file_ending
            if not return_peak_count:
                self.save((X,Y), saving_path, file_name)
            else:
                file_name += '_with_peaks'
                self.save((X,Y,P), saving_path, file_name )
            # add to .gitignore
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

class WavelengthRangeError(Exception):
    """ An exception that is raised when wavelength ranges dont match. """
    pass

def merge_dictionary_contents(dict1, dict2):
    merged_dict = dict1
    for key, value in dict2.items():
        if key in merged_dict.keys():
            merged_dict[key] = np.vstack((merged_dict[key], value))
        else:
            merged_dict[key] = value
    return merged_dict
            