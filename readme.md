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

Pre-trained autoencoder weights for processing of spectra recorded over a wavelength range of 30 micrometers already exist in the [models](/models/autoencoders/) directory. For retraining however, e.g. for processing another class of spectra, a respective Juypter Notebook [train_autoencoder.ipynb](/notebooks/train_autoencoder.ipynb) is provided, together with a [visualizing evaluation notebook](/notebooks/evaluate_training.ipynb). The notebook is constructed for the use in [Google Colab](https://colab.research.google.com/?hl=de). Note that the network structure allways mirrors [encoder.py](/src/lib/neuralNetworks/encoder.py) and [decoder.py](/src/lib/neuralNetworks/decoder.py). 

The autoencoder can be exploited for feature learning or feature construction. For example, the anomaly score (or the reconstruction error in other words) of the Spectrum is highly correlating with the Spectrum label.

<img src="/reports/graphics/correlation_label_vs_anomaly_score.PNG">

Also, the latent representation of the spectrum can be used as feature vector for a subsequent regression model input. For the optimal latent space search, a script [autoencoder_selection.py](/src/scripts/autoencoder_selection.py) is provided. 


### Feature Construction and Selection

As mentioned, the anomaly score can be exploited as powerful feature. To construct more features, [peak detection algorithms](/src/lib/peakDetectors/) are implemented within the scope of this library. 

<img src="/reports/graphics/peak_detection.PNG">

An exemplary parameter optimization based on the labeled dataset for the popular [OS-CFAR detector](/src/lib/peakDetectors/os_cfar.py) peak detector is provided in the script [optimize_CFAR.py](/src/scripts/optimize_CFAR.py).

Based on the peak detector and the autoencoder, several intuition based features can be derived. The [FeatureExtractor class](/src/lib/featureExtraction/feature_extractor.py) implements two feature selection algorithms based on a k-Fold cross validation and a neural network based minimal regressor, which is demonstrated in the [feature_selection.py](/src/scripts/feature_selection.py) script.

<img src="/reports/graphics/feature_selection.PNG">


### Probabilistic Regression

For the final model learning, a self-normalizing neural-network-based stochastic regression model is used. Analogous to Gaussian Processes, this network is able to predict labels together with a normal distribution based standart deviation. A multivariate [regression script](/src/scripts/feature_regression.py) is provided, the settings can be adjusted within a [configuration YAML file](/src/scripts/config/config.yaml). The script is executed with the syntax

	python3 src/scripts/feature_regression.py --config src/scripts/config/config.yaml

<img src="/reports/graphics/stochastic_regression.PNG">


### Documentation

The corresponding publication explaining the algorithms and the citation will be announced here upon it's publishment.


### Acknoledgement

This project has been conducted as a student work enabled by the research project [GRK 2642: "Towards Graduate Experts in Photonic Quantum Technologies"](https://www.pqe.uni-stuttgart.de/).
