import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Mish

class Model(tf.keras.Model):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(height, width, in_channels))
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def call(self, x):
        x = self.conv(x)
        x = x - self.subtract_value_1
        x = x - self.subtract_value_2
        x = Mish()(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
