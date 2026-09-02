import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding='same', output_padding=output_padding)
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.scaling_factor = scaling_factor

    def call(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        x = x * self.scaling_factor
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        x = x / self.scaling_factor
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
