import tensorflow as tf
from tensorflow.keras.layers import Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        """
        Applies ReLU activation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of any shape.

        Returns:
            tf.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return Activation('relu')(inputs)

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.normal([batch_size, dim])
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
