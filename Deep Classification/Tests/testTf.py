"""
Tests installation of tensorflow and NVIDIA GPU support
"""
import tensorflow as tf

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.config.list_physical_devices('GPU'))