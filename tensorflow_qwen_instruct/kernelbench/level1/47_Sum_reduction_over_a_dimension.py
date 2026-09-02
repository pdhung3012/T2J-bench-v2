import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def call(self, inputs):
        """
        Applies sum reduction over the specified dimension.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, dim1, dim2).

        Returns:
            tf.Tensor: Output tensor after sum reduction, shape (batch_size, 1, dim2).
        """
        return tf.reduce_sum(inputs, axis=self.dim, keepdims=True)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    x = tf.random.normal((batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [reduce_dim]
