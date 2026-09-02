import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GroupNormalization, Sigmoid

class Model(tf.keras.Model):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.scale = tf.Variable(tf.random.normal(scale_shape))
        self.group_norm = GroupNormalization(groups=num_groups)

    def call(self, x):
        x = self.conv(x)
        x = x + self.bias[None, :, None, None]
        x = x * self.scale[None, :, None, None]
        x = Sigmoid()(x)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [tf.random.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
