import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')

    def call(self, x):
        x = self.conv(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
