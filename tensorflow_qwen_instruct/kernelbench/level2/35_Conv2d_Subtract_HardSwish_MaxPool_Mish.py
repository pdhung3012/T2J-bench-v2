import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Mish

class Model(tf.keras.Model):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size=(kernel_size, kernel_size), input_shape=(in_channels, height, width))
        self.subtract_value = subtract_value
        self.pool = MaxPooling2D(pool_size=(pool_kernel_size, pool_kernel_size))

    def call(self, inputs):
        x = self.conv(inputs)
        x = x - self.subtract_value
        x = tf.nn.hard_swish(x)
        x = self.pool(x)
        x = Activation('mish')(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
