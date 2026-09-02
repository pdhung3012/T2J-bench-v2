import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def call(self, inputs):
        x, mask = inputs
        return tf.cumsum(x * mask, axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    x = tf.random.uniform((batch_size, *input_shape))
    mask = tf.cast(tf.random.int32((batch_size, *input_shape)) % 2, tf.bool)  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]
