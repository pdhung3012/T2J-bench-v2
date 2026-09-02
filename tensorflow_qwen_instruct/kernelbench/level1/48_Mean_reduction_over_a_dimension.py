import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

class Model(tf.keras.Model):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (tf.Tensor): Input tensor of arbitrary shape.

        Returns:
            tf.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return tf.reduce_mean(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = tf.random.normal((batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [1]
