import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

class Model(tf.keras.Model):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.dense = Dense(output_size, input_shape=(input_size,))
        self.divisor = divisor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.dense(x)
        x = x / self.divisor
        x = Activation('gelu')(x)
        return x

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, output_size, divisor]
