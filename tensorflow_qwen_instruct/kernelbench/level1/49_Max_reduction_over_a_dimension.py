import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def call(self, inputs):
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return tf.reduce_max(inputs, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = tf.random.uniform((batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension
