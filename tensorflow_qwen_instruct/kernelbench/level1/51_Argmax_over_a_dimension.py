import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def call(self, x):
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return tf.argmax(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = tf.random.uniform((batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1]
