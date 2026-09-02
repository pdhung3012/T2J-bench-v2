import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation

class Model(tf.keras.Model):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')
        self.multiplier = tf.Variable(tf.random.normal(multiplier_shape))
        self.leaky_relu = LeakyReLU()
    
    def call(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = tf.nn.gelu(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
