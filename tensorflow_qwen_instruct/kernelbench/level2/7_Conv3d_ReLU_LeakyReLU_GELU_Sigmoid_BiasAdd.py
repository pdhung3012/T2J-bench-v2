import tensorflow as tf
from tensorflow.keras.layers import Conv3D, ReLU, LeakyReLU, Gelu, Sigmoid

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, input_shape=(depth, height, width, in_channels))
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv(x)
        x = ReLU()(x)
        x = LeakyReLU(negative_slope=0.01)(x)
        x = Gelu()(x)
        x = Sigmoid()(x)
        x = x + self.bias
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
