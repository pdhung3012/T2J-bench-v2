import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def call(self, x):
        cumsum = tf.cumsum(x.narrow(self.dim, 0, tf.size(x) - 1), axis=self.dim)
        return tf.concat((tf.zeros_like(x.select(self.dim, 0).expand(self.dim)), cumsum), axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [tf.random.uniform(shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
