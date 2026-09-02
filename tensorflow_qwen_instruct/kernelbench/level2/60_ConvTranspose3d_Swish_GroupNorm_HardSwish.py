import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, GroupNormalization, Activation

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding, use_bias=bias)
        self.group_norm = GroupNormalization(groups=groups, epsilon=eps)

    def call(self, x):
        x = self.conv_transpose(x)
        x = tf.math.sigmoid(x) * x  # Swish activation
        x = self.group_norm(x)
        x = tf.nn.hardswish(x)  # HardSwish activation
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]
