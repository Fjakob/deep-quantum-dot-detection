from __config__ import *

from src.lib.dataHandling.data_processing import DataProcesser


if __name__ == "__main__":

    db_abs_path = 'C:\\Users\\Fabian\\Documents\\Maps_for_ISYS'

    loader = DataProcesser(db_abs_path, spectrum_size=1024)

    #dataset = loader.load_spectra_from_database(saving_path='datasets/raw')
    dataset = loader.load_spectra_from_file(file_path='datasets/raw/database.pickle')
    X = loader.preprocess(dataset, w_range=12, background_filtering=True)
    X = loader.augment(X, space_shifts=2, mirroring=False, saving_path='datasets/unlabeled')

