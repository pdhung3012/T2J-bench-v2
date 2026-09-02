import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = x * self.multiplier
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # First global average pooling
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # Second global average pooling
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]
