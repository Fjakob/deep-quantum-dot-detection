# DQDD - Deep Quantum Dot Detection

### First step

To install all dependencies, run
	
	pip install -r requirements.txt


### Data set creation

Requires:

1.) A database folder on the PC containing of the form

	|-- folder1
	|	|-- filename1.dat     
	|	|-- filename2.dat
	|	|-- ...
	|
	|-- folder2
	|	|-- ...
	|
	|-- ...

2.) A folder containing labeled spectra in the directory `datasets/labeled/labels_txt/`

Entries in that txt files should have following form:

label `num_peaks` `rating` date `date` user `user` file `folder/filename`

txt files of this form can be created by using the Labeling App in `src/apps/`

The datasets can be created by using the script `src/scripts/create_datasets.py`
by using the `DataProcesser.create_regression_data(...)` method.
See comments in `src/lib/dataProcessing/data_processer.py`.

Datasets get automatically saved into `datasets/labeled/` as pickle file.

### Autoencoder Training

The datasets can be created by using the script `src/scripts/create_datasets.py` by using the `DataProcesser.create_unsupervised_data(...)` method.
See the comments in `src/lib/dataProcessing/data_processer.py`

For training: Google colab notebook exists in `notebooks/train_autoencoder.ipynb`
which can be uploaded to colab.research.google.com. 
In the colab, upload the dataset created in the last step (might take some time.)

In the notebook, network structures and latent dimensions can be specified.
After training, the model parameters autoencoder.pth and learning curves will be downloaded.

Downloaded training curves must be moved to folder `reports/autoencoder_training_curves/`.
Training curves can be evaluated in `notebooks/evaluate_training.ipynb`. 

For deploying the autoencoder parameters in the library,m ake sure the network structure chosen in `train_autoencoder.py` matches the structures in `src/lib/neuralNetworks/(encoder/decoder).py` and `src/lib/featureExtraction/autoencoder.py`.

Move downloaded model parameters to folder `models/autoencoders/`.

Examples how to load model parameters are given in `src/scripts/autoencoder_selection.py`


### LIBRARY USAGE

The whole library containing algorithms is contained in `src/lib/`.

Examples on how to use the library are given in the scripts `src/scripts`.


### SCRIPT USAGE

All scripts except for the regression scripts can be executed directly. Make sure 
to execute all scripts while the root folder is the folder in which
the whole project is contained. Visual Studio Code is suggested.

To execute regression scripts, the config.yaml file has to be passed. Therefore, run 
the script in the root folder using the command line like this:

	python3 src/scripts/erecon_regression.py --config=src/scripts/config/config.yaml
	python3 src/scripts/feature_regression.py --config=src/scripts/config/config.yaml
	python3 src/scripts/feature_regression_augmented.py --config=src/scripts/config/config.yaml

The parameters of every pipeline is unifiedly set and can be changed in `src/scripts/config/config.yaml`

### TESTING 

Python unit tests have been included to test the library functionality in `tests/`

However, test cases for the whole library yet have to be developed for proper code coverage.


### THEORY BEHIND

The underlying Master Thesis will be published eventually and linked here.


   
