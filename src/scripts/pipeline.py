# OUTLINE:
#
# i)  database of spectra recorded over different wavelenghts (sample size around 1k - 20k)
# ii) labeled data of spectra on one wavelength range (sample size around 100)


###############################################################################################
#       S U P E R V I S E D   A P P R O A C H E S
###############################################################################################

# PIPELINE I: intuitive feature extraction
#
# 1) extract features with peak detection (number of peaks, maximum width, minimum distance, etc)
# 2) use features and labeled data to learn a regressor (e.g. shallow neural network or gaussian process)
# 
# + intuitive, explainable
# + fast, easy implementation
# - very small amount of labeled data, bad coverage of input space (ca 5-7 dim.)
# - data not augmentable, since peak detection is space shift invariant
#  
# expected: very bad accuracy


# PIPELINE II: direct deep learning regression
# 
# 1) normalize and augment (space shift, mirror) labeled data
# 2) learn a deep learning regression model
#
# + automated spatial feature extraction
# - augmentation possible, but might only enhance overfitting
# - no intuition of neural net architecture
# - not interpretable 
# 
# expected: overfitting, maybe no convergence


###############################################################################################
#       U N S U P E R V I S E D   E N H A N C E D   A P P R O A C H E S
###############################################################################################

# PIPELINE III: autoencoder feature based
# 
# 1) load whole unlabeled database, filter, normalize and augment
# 2) train autoencoder in unsupervised manner
# 3) normalize and augment (space shift, mirror) labeled data
# 4) extract features with autoencoder (e.g. 20-dim embedding vector)
# 4) use features and labeled data to learn a regressor (e.g. shallow neural network or gaussian process)
# 
# + representation learning with big data coverage
# + very well augmentable
# - input space still huge (ca 20 dim.), data probably still not enough for coverage of whole input space
# - very cumbersome (umständlich)
# - not interpretable
#
# observed: some peaks not well reconstructed => cutoff of relevant features
# expected: bad regression results due to feature cutoff, bad prediction results for uncovered data


# PIPELINE IV: autoencoder reconstruction based
# 
# 1) load whole unlabeled database, filter, normalize and augment
# 2) train autoencoder in unsupervised manner
# 3) use reconstruction error as a measure of how bad the spectra is (good unnoisy spectra can be reconstructed well)
# 4) use a neural network to map reconstruction error to label
# 
# + interpretable, intuitive
# + input space very small (1-dim) => good coverage and generalization
# - not augmentable since reconstruction error is space shift invariant
# - cumbersome (umständlich)
#
# expected: promising, but highly dependent on autoencoder performance



# DEPLOYEMENT:
# 
# 1) Know the wavelength range you want to investigate
# 2) Label around 100 samples of spectra of that wavelength range with 6-7 experts to have good mean labels for regression
# 3) Perform one of PIPELINE I-IV 
# 4) Exploit regressor on a given semiconductor sample, label every point on the sample
# 5) Return coordinates of the point with best ratings
# 6) Use peak detection algorithm to return peak height and wavelength
