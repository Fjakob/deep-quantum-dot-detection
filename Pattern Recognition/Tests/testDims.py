import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D

input_shape = (1,1024,1)
x = tf.random.normal(input_shape)
x = Conv1D(8, 32, activation='relu', padding='same')(x)
print(x.shape)
x = Conv1D(4, 16, activation='relu', padding='same')(x)
print(x.shape)
x = Conv1D(1, 8, activation='relu', padding='same')(x)

