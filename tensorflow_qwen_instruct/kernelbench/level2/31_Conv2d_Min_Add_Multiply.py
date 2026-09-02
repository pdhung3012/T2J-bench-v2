import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(in_channels, height, width))
        self.constant_value = constant_value
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.scaling_factor = scaling_factor

    def call(self, x):
        x = self.conv(x)
        x = tf.minimum(x, tf.ones_like(x) * self.constant_value)
        x = x + self.bias
        x = x * self.scaling_factor
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [tf.random.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
