import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import pickle


class SpectrumRater(keras.Model):
    def __init__(self):
        super(SpectrumRater, self).__init__()
        self.conv0   = keras.layers.Conv1D(8, 9, strides=4, activation="selu")
        self.conv1   = keras.layers.Conv1D(16, 9, strides=4, activation="selu")
        self.conv2   = keras.layers.Conv1D(32, 9, activation='selu')
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(rate=0.2)
        self.dense0  = keras.layers.Dense(128, activation="selu")
        self.dense1  = keras.layers.Dense(64, activation="selu")
        self.dense2  = keras.layers.Dense(5, activation="softmax")

    def call(self, inputs, training=False):
        output = self.conv0(inputs)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.dense0(output)
        output = self.dropout(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output


if __name__ == '__main__':

    # Load Dataset
    with open('dataSet', 'rb') as f:
        dataSet = pickle.load(f)
    for w, x, label in dataSet:
        y = label[1]
        try:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
        except(NameError):
            X = x
            Y = y

    n_dim  = X.shape[1]
    n_data = X.shape[0]
    n_train = int(0.9*n_data)

    # Shuffle the data
    shuffle_idx = np.arange(0, n_data)
    np.random.shuffle(shuffle_idx)
    # Divide into train, test, validation
    X = np.expand_dims(X, axis=-1)
    X_shuffled = X[shuffle_idx]
    Y_shuffled = Y[shuffle_idx] + 2
    X_train = X_shuffled[0:n_train]
    Y_train = Y_shuffled[0:n_train]
    X_test = X_shuffled[n_train:]
    Y_test = Y_shuffled[n_train:]

    # Model Hyperparameters
    batch_size = 16
    epochs = 100
    learning_rate = 0.001

    # Create Model
    mdl = SpectrumRater()
    opt = tf.keras.optimizers.RMSprop(learning_rate)
    mdl.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train model
    history = mdl.fit(X_train, Y_train, batch_size, epochs, validation_split=0.1)

    mdl.summary()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.xticks(range(epochs))
    plt.xlabel("epochs")
    plt.title("Training process")
    plt.show()

    # Test model
    for idx in range(10):
        y_pred = mdl(np.expand_dims(X_test[idx],axis=0))
        y_real = Y_test[idx]
        plt.plot(X_test[idx])
        plt.title("Real Rating: {0}\nPredicted Rating: {1} (P(y) = {2:.2g})".format(int(y_real[0])-2, np.argmax(y_pred[0])-2, np.max(y_pred[0])))
        plt.show()

