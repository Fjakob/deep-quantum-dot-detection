from config.__config__ import *

from src.lib.dataProcessing.data_processer import DataProcesser


def main():
    """ Script to create datasets from database. """

    ### Setup
    w_range, spectrum_length = 30, 1024
    db_abs_path = 'C:\\Users\\Fabian\\Documents\\Maps_for_ISYS'
    loader = DataProcesser(db_abs_path, spectrum_size=spectrum_length)

    ### 1) Create Data for Unsupervised Learning of Autoencoder
    #loader.create_unsupervised_data(w_range=w_range, background_filtering=True, augment=True, 
     #                                loading_path='datasets/raw/database.pickle', saving_path='datasets/unlabeled')

    ###  2) Create Data for Supervised Learning of Classifier
    txt_dir = 'datasets\\labeled\\labels_txt'
    loader.create_regression_data(w_range, txt_dir, return_peak_count=False, 
                                        saving_path='datasets/labeled', augment=False)


if __name__ == "__main__":
    main()