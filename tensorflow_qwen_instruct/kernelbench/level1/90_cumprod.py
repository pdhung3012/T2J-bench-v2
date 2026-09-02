import tensorflow as tf
from tensorflow.keras.layers import Layer

class CumulativeProductModel(Layer):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(CumulativeProductModel, self).__init__()
        self.dim = dim

    def call(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            tf.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return tf.math.cumprod(x, axis=self.dim)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [tf.random.uniform((batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
