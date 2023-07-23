# DQDD - Deep Quantum Dot Detection

DQDD - A python library for easy and convenient automation of quantum dot fabrication. 

## Background

Quantum dots (QDs) are emerging as a key technology to realize Quantum Computing, Quantum Communication, and many other quantum application technologies. Efficient and scalable fabrication will thus become key for solving challenging 21st century tasks in this industry. This library aims at automating the assessment of individual quantum dots in a precise, quantifiable and reproducable way with the help of advanced evaluation methods and Deep Learning. 

<img src="/reports/graphics/example_QDs.PNG">


## Installation and Requirements

To install all dependencies, run
	
	pip install -r requirements.txt

To speed up your device, torch CUDA extensions can be optionally downloaded. For this, check your CUDA-Version by running

	nvidia-smi

in the command line. Then follow the [torch installation guidelines](https://pytorch.org/) from the website.


## Library Usage

This library implements a whole end-to-end machine learning pipeline, beginning with raw QD spectral data up to a deployable stochastic prediction model. The main features include

* Data labeling with an interactive App
* Automated feature leaning with an off-the-shelf Autoencoder
* Manual feature design by implemented CFAR detectors and selection algorithms
* Regression model learning with a stochastic neural network 

<img src="/reports/graphics/library_concept_drawing.PNG">

### Data labeling

Outline is a raw database, which is a directory of raw QD spectral measurements, saved as `txt` or `dat` files, whereas the spectra should have been recorded over the same frequency range.

The [apps folder](/src/apps/) contains different versions of a labeling GUI to create datasets for supervised learning. The interface allows to select folders within the OS explorer, in example the raw database, and saves the spectra with all information in a resulting `txt` file, which can be saved in the [labels_txt](/datasets/labeled/labels_txt/) directory for later usage. The versions include

| Version  | Description |
| ------------- | ------------- |
| v3  | Categorical rating of deterministic selected spectra within an opened directory |
| v4  | Categorical rating of randomly selected spectra within a directory and all subdirectories |
| v5  | Continuous rating of randomly selected spectra within a directory and all subdirectories |

The GUI allows for

* enter of a categorical rating in the range from (--) to (++) in v3/v4 or as continuous value in the range (0, 1) in v5
* enter of the number of Peaks visible in that spectrum.

<img src="/reports/graphics/label_app.PNG">

Both information are exploited in the following algorithms.

Once sufficiently many spectra have been assessed and saved into the [labels_txt](/datasets/labeled/labels_txt/) directory, the [create_datasets.py](/src/scripts/create_datasets.py) script can be exploited to create the respective numpied pickle objects representing all data pairs by reading out the raw data base. Additionally, the [create_datasets.py](/src/scripts/create_datasets.py) can be used to create a dataset for unsupervised learning, which is relevant for the Autoencoder training for example. Features of the [DataProcesser](/src/lib/dataProcessing/data_processer.py) class include

* readout of the raw database, sorting for wavelength ranges
* noise and peak filtering of each sample
* data augmentation by shifting and mirroring

An example usage of the class is presented in the [test_dataloader unittest](/tests/test_dataloader.py).


### Autoencoder Training

Pre-trained autoencoder weights for spectra recorded over a wavelength range of 30 micrometers already exist in the [models](/models/autoencoders/) directory. However, for retraining, a respective Juypter Notebook [train_autoencoder.ipynb](/notebooks/train_autoencoder.ipynb) is provided, together with a [visualizing evaluation notebook](/notebooks/evaluate_training.ipynb).

The notebook should be uploaded to [colab.research.google.com](colab.research.google.com). 
In the colab, upload the dataset created in the last step (this might take some time).

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


### Documentation

The corresponding publication explaining the algorithms and the citation will be announced here upon it's publishment.


### Acknoledgement

This project has been conducted as a student work enabled by the research project [GRK 2642: "Towards Graduate Experts in Photonic Quantum Technologies"](https://www.pqe.uni-stuttgart.de/).
