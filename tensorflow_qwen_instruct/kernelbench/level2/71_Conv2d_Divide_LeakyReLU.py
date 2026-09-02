import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(in_channels, height, width))
        self.divisor = divisor

    def call(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = LeakyReLU(negative_slope=0.01)(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
