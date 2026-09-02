import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, input_shape=(depth, height, width, in_channels))
        self.scaling_factor = tf.Variable(tf.random.normal(bias_shape))
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv(x)
        x = x * self.scaling_factor 
        x = tf.nn.tanh(x)
        x = x * self.bias
        x = tf.nn.sigmoid(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
