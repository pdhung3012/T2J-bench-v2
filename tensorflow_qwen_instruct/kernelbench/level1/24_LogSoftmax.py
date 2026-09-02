import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim
    
    def call(self, x):
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            tf.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return tf.math.log(tf.nn.softmax(x, axis=self.dim))

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.uniform((batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
