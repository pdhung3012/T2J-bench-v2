import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class Model(tf.keras.Model):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def call(self, inputs):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of any shape.

        Returns:
            tf.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return LeakyReLU(negative_slope=self.negative_slope)(inputs)

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.normal([batch_size, dim])
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
