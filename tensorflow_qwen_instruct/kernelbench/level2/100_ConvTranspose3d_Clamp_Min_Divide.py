import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose

class Model(tf.keras.Model):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same')
        self.min_value = min_value
        self.divisor = divisor

    def call(self, x):
        x = self.conv_transpose(x)
        x = tf.clip_by_value(x, clip_value_min=self.min_value)
        x = x / self.divisor
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
