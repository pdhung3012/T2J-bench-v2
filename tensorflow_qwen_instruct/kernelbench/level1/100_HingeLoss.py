import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np

class Model(tf.keras.Model):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.dense = Dense(dim, input_shape=input_shape)

    def call(self, inputs):
        x, y = inputs
        z = self.dense(x)
        return tf.reduce_mean(tf.maximum(0., 1. - z * y))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [tf.random.uniform(shape=(batch_size, *input_shape)), tf.random.uniform(shape=(batch_size,), minval=-1, maxval=1)]

def get_init_inputs():
    return []
