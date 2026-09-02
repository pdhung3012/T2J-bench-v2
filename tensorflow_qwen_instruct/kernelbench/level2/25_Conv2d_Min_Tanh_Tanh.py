import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MinMaxLayer, Activation

class Model(tf.keras.Model):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(in_channels, height, width))

    def call(self, x):
        x = self.conv(x)
        x = tf.reduce_min(x, axis=1, keepdims=True)  # Apply minimum operation along the channel dimension
        x = Activation('tanh')(x)
        x = Activation('tanh')(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
