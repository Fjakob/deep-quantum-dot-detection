import numpy as np
import matplotlib.pyplot as plt
# Classifiers:
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# For data preprocessing:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# For visualization:
from sklearn.inspection import DecisionBoundaryDisplay

from FeatureExtraction import extractFeatures
from FeatureExtraction import loadDataSet

class Classifier():
    def __init__(self, method):
        methods = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]
        classifiers = [
            KNeighborsClassifier(3), # k nearest neighbours
            LinearSVC(multi_class="crammer_singer"),
            SVC(kernel="rbf", decision_function_shape='ovo', gamma=2, C=1), # support vector machine with RBF kernel
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]        
        self.classifier = classifiers[methods.index(method)]
    
    def to_features(self, X):
        for x in X:
            v = np.asarray(list(extractFeatures(x).values()))
            try:
                V = np.vstack((V, v))
            except(NameError):
                V = v
        return V
    
    def train(self, X, Y):
        # Project into feature space and scale data
        V = self.to_features(X)
        V = StandardScaler().fit_transform(V)
        # Split data to train and test dataset
        V_train, V_test, y_train, y_test = train_test_split(V, Y, test_size=0.4, random_state=42)
        # Train the model
        self.classifier.fit(V_train, y_train)
        # Test the model
        score = self.classifier.score(V_test, y_test)
        print("Test Accuracy: {0}".format(score))
    
    def __call__(self, x):
        v = self.to_features(x)
        y = self.classifier.predict([v])
        return y
    
    
if __name__ == '__main__':

    # load data set
    X, Y = loadDataSet('dataSet')
    Y = np.ravel(Y)
    
    classifier = Classifier("Linear SVM")
    classifier.train(X,Y)

    for idx in range(len(X)):
        x = X[idx]
        y = classifier([x])
        print("Real: {}, Predicted: {}".format(Y[idx], y[0]))




    